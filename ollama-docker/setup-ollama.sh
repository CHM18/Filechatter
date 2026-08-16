#!/bin/bash
set -e

CONTAINER_NAME="ollama-container"
IMAGE_NAME="ollama:latest"
OLLAMA_DIR="${HOME}/ollama"
MODEL_NAME="nemotron-3-nano:4b"

# ── Create model storage directory ────────────────────────────────
if [ ! -d "$OLLAMA_DIR" ]; then
    echo "→ Creating model storage directory at $OLLAMA_DIR ..."
    mkdir -p "$OLLAMA_DIR"
fi

# ── Stop and remove existing container if it exists ───────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "→ Stopping existing container '${CONTAINER_NAME}' ..."
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
fi

# ── Build the Docker image ───────────────────────────────────────
echo "→ Building Docker image '${IMAGE_NAME}' ..."
docker build -t "$IMAGE_NAME" "$(dirname "$0")"

# ── Install NVIDIA Container Toolkit if missing ───────────────────
install_nvidia_container_toolkit() {
    echo "→ Installing NVIDIA Container Toolkit ..."
    if command -v apt-get &>/dev/null; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
    elif command -v dnf &>/dev/null; then
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
        sudo dnf install -y nvidia-container-toolkit
    elif command -v yum &>/dev/null; then
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
        sudo yum install -y nvidia-container-toolkit
    else
        echo "⚠  No supported package manager (apt/dnf/yum) found — cannot auto-install."
        echo "   Follow the manual install guide:"
        echo "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        return 1
    fi

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null || true
}

# ── Detect NVIDIA GPU ────────────────────────────────────────────
GPU_FLAGS=""
CDI_SPEC_PATH="/etc/cdi/nvidia.yaml"
CDI_GENERATE_ARGS="--output=${CDI_SPEC_PATH}"
if grep -qi microsoft /proc/version 2>/dev/null; then
    # WSL2 needs --mode=wsl so the ldcache hook targets the driver-store folder.
    CDI_GENERATE_ARGS="--mode=wsl ${CDI_GENERATE_ARGS}"
fi

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "→ NVIDIA GPU detected on host — preparing CDI spec ..."

    if ! command -v nvidia-ctk &>/dev/null; then
        install_nvidia_container_toolkit || true
    fi

    # Always (re)generate so the spec reflects the correct mode (e.g. WSL) and current driver.
    if command -v nvidia-ctk &>/dev/null; then
        echo "→ Generating CDI spec at ${CDI_SPEC_PATH} (args: ${CDI_GENERATE_ARGS}) ..."
        sudo mkdir -p "$(dirname "$CDI_SPEC_PATH")"
        sudo nvidia-ctk cdi generate $CDI_GENERATE_ARGS
    else
        echo "⚠  nvidia-ctk still not available — cannot generate a CDI spec."
        echo "   Install the NVIDIA Container Toolkit manually:"
        echo "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi

    # Verify --gpus all actually works end-to-end before trusting it for the real container.
    # --entrypoint overrides the image's "ollama" entrypoint so nvidia-smi actually runs.
    echo "→ Verifying Docker GPU access ..."
    GPU_TEST_OUTPUT="$(timeout 30 docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_NAME" 2>&1)"
    GPU_TEST_STATUS=$?
    if [ $GPU_TEST_STATUS -eq 0 ]; then
        echo "✓ Docker GPU access confirmed — enabling GPU support ..."
        GPU_FLAGS="--gpus all"
    else
        if [ $GPU_TEST_STATUS -eq 124 ]; then
            echo "⚠  GPU test timed out after 30s. Running in CPU-only mode."
        else
            echo "⚠  Docker could not access the GPU (--gpus all test failed). Running in CPU-only mode."
        fi
        echo "   Test output:"
        echo "$GPU_TEST_OUTPUT" | sed 's/^/     /'
        echo "   Common causes: CDI spec missing/stale, NVIDIA Container Toolkit not configured,"
        echo "   or (on Docker Desktop/WSL2) GPU support not enabled in Docker Desktop settings."
        echo "   Docs: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
else
    echo "⚠  nvidia-smi not found or no GPU detected. Running in CPU-only mode."
    echo "   For GPU support, install:"
    echo "     1. NVIDIA drivers: https://www.nvidia.com/drivers"
    echo "     2. NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

# ── Start the container ──────────────────────────────────────────
echo "→ Starting container '${CONTAINER_NAME}' ..."
docker run -d \
    $GPU_FLAGS \
    --name "$CONTAINER_NAME" \
    -v "${OLLAMA_DIR}:/root/.ollama" \
    -p 11434:11434 \
    --restart unless-stopped \
    "$IMAGE_NAME"

# ── Wait for Ollama to be ready ──────────────────────────────────
echo ""
echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 >/dev/null 2>&1; then
        echo "✓ Ollama is ready!"
        break
    fi
    sleep 2
done

# ── Confirm GPU is actually in use inside the running container ──
if [ -n "$GPU_FLAGS" ]; then
    echo ""
    echo "→ Confirming GPU is active inside '${CONTAINER_NAME}' ..."
    if docker exec "$CONTAINER_NAME" nvidia-smi &>/dev/null; then
        echo "✓ GPU is active inside the container."
    else
        echo "⚠  GPU flags were passed but nvidia-smi failed inside the running container."
        echo "   The model will likely run on CPU. Check 'docker logs ${CONTAINER_NAME}' for details."
    fi
fi

# ── Pull the requested model ─────────────────────────────────────
echo ""
echo "→ Pulling model '${MODEL_NAME}' ..."
docker exec -it "$CONTAINER_NAME" ollama pull "$MODEL_NAME"

# ── Start interactive chat ───────────────────────────────────────
echo ""
echo "→ Starting interactive chat with '${MODEL_NAME}' ..."
echo "   Type your messages and press Enter. (Ctrl+C to exit)"
echo ""
docker exec -it "$CONTAINER_NAME" ollama run "$MODEL_NAME"
