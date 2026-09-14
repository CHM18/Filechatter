#!/bin/bash
set -e

CONTAINER_NAME="ollama-container"
IMAGE_NAME="ollama:latest"
OLLAMA_DIR="${HOME}/ollama"
MODEL_NAME="nemotron-3-nano:4b"

install_docker_if_missing() {
    if command -v docker &>/dev/null; then
        return 0
    fi

    echo "→ Docker not found. Installing Docker ..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update
        sudo apt-get install -y apt-utils docker.io
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y docker
    elif command -v yum &>/dev/null; then
        sudo yum install -y docker
    else
        echo "✗ Unsupported package manager. Please install Docker manually:"
        echo "  https://docs.docker.com/engine/install/"
        return 1
    fi
}

ensure_docker_ready() {
    install_docker_if_missing || return 1

    # Start daemon if it is not running yet.
    if ! docker info >/dev/null 2>&1; then
        sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || true
    fi

    if docker info >/dev/null 2>&1; then
        return 0
    fi

    echo "✗ Docker is installed but not usable for the current user."
    echo "  Try one of the following and rerun this script:"
    echo "    1) sudo usermod -aG docker ${USER}"
    echo "    2) Log out/in (or run: newgrp docker)"
    echo "    3) Ensure docker daemon is running"
    return 1
}

ensure_docker_ready

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

verify_nvidia_container_toolkit() {
    echo "→ Verifying NVIDIA Container Toolkit installation ..."
    if ! command -v nvidia-ctk &>/dev/null; then
        echo "⚠  nvidia-ctk is not installed or not in PATH."
        return 1
    fi
    if ! command -v nvidia-container-cli &>/dev/null; then
        echo "⚠  nvidia-container-cli is missing (toolkit installation looks incomplete)."
        return 1
    fi
    if ! nvidia-ctk --version &>/dev/null; then
        echo "⚠  nvidia-ctk is installed but not runnable."
        return 1
    fi
    echo "✓ NVIDIA Container Toolkit is installed."
    return 0
}

set_nvidia_runtime_mode() {
    local mode="$1"
    if ! command -v nvidia-ctk &>/dev/null; then
        return 1
    fi
    sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode="${mode}" || return 1
    sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null || true
    return 0
}

prepare_wsl_legacy_libs() {
    local wsl_lib_dir="/usr/lib/wsl/lib"
    local host_lib_dir="/usr/lib/x86_64-linux-gnu"

    if [ ! -d "$wsl_lib_dir" ]; then
        return 1
    fi

    echo "→ Staging WSL NVIDIA libraries for legacy runtime mode ..."
    for lib in libnvidia-ml.so.1 libnvidia-ml.so libcuda.so.1 libcuda.so; do
        if [ -e "${wsl_lib_dir}/${lib}" ]; then
            sudo install -m 755 "${wsl_lib_dir}/${lib}" "${host_lib_dir}/${lib}"
        fi
    done
    sudo ldconfig || true
    return 0
}

# ── Detect NVIDIA GPU ────────────────────────────────────────────
GPU_FLAGS=""
CDI_SPEC_PATH="/etc/cdi/nvidia.yaml"
CDI_GENERATE_ARGS="--output=${CDI_SPEC_PATH}"
CDI_SPEC_BACKUP_PATH="${CDI_SPEC_PATH}.filechatter.bak"
WSL_LIBDXCORE_PATH="/usr/lib/wsl/lib/libdxcore.so"
IS_WSL=0
USE_WSL_CDI=0
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
fi
if [ "$IS_WSL" -eq 1 ] && [ -e "$WSL_LIBDXCORE_PATH" ]; then
    # WSL2 with mounted lib path present: use WSL mode for CDI generation.
    USE_WSL_CDI=1
    CDI_GENERATE_ARGS="--mode=wsl ${CDI_GENERATE_ARGS}"
elif [ "$IS_WSL" -eq 1 ]; then
    echo "⚠  WSL detected but ${WSL_LIBDXCORE_PATH} was not found."
    echo "   Falling back to non-WSL CDI generation to avoid broken GPU mounts."
fi

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "→ NVIDIA GPU detected on host — preparing CDI spec ..."

    if ! verify_nvidia_container_toolkit; then
        install_nvidia_container_toolkit || true
    fi

    # Always (re)generate so the spec reflects the correct mode (e.g. WSL) and current driver.
    if verify_nvidia_container_toolkit; then
        echo "→ Generating CDI spec at ${CDI_SPEC_PATH} (args: ${CDI_GENERATE_ARGS}) ..."
        sudo mkdir -p "$(dirname "$CDI_SPEC_PATH")"
        sudo nvidia-ctk cdi generate $CDI_GENERATE_ARGS

        # Verify --gpus all actually works end-to-end before trusting it for the real container.
        # --entrypoint overrides the image's "ollama" entrypoint so nvidia-smi actually runs.
        echo "→ Verifying Docker GPU access ..."
        set +e
        GPU_TEST_OUTPUT="$(timeout 30 docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_NAME" 2>&1)"
        GPU_TEST_STATUS=$?
        set -e

        if [ $GPU_TEST_STATUS -eq 0 ]; then
            echo "✓ Docker GPU access confirmed — enabling GPU support ..."
            GPU_FLAGS="--gpus all"
        else
            if [ $GPU_TEST_STATUS -eq 124 ]; then
                echo "⚠  GPU test timed out after 30s. Running in CPU-only mode."
            else
                echo "⚠  Docker could not access the GPU (--gpus all test failed). Running in CPU-only mode."

                # Some WSL environments report 'microsoft' in /proc/version but do not expose
                # /usr/lib/wsl/lib/libdxcore.so. In that case, retry once with a non-WSL CDI spec.
                if [ "$USE_WSL_CDI" -eq 1 ] && echo "$GPU_TEST_OUTPUT" | grep -q "libdxcore.so"; then
                    echo "→ Retrying with fallback CDI spec (without --mode=wsl) ..."
                    sudo nvidia-ctk cdi generate --output="${CDI_SPEC_PATH}"
                    set +e
                    GPU_TEST_OUTPUT="$(timeout 30 docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_NAME" 2>&1)"
                    GPU_TEST_STATUS=$?
                    set -e
                    if [ $GPU_TEST_STATUS -eq 0 ]; then
                        echo "✓ Docker GPU access confirmed after fallback CDI regeneration."
                        GPU_FLAGS="--gpus all"
                    else
                        echo "⚠  Fallback CDI retry also failed. Running in CPU-only mode."
                    fi
                fi

                # Some WSL + Docker setups expose host files but still fail CDI mount fulfillment
                # during OCI init. Retry once by disabling CDI spec discovery so runtime falls back
                # to non-CDI GPU discovery mode.
                if [ $GPU_TEST_STATUS -ne 0 ] && echo "$GPU_TEST_OUTPUT" | grep -q "failed to fulfil mount request" && echo "$GPU_TEST_OUTPUT" | grep -q "libdxcore.so"; then
                    echo "→ Detected libdxcore CDI mount fulfillment failure."
                    echo "→ Retrying with Docker runtime mode set to 'auto' (legacy GPU discovery) ..."
                    if sudo test -f "$CDI_SPEC_PATH"; then
                        sudo mv "$CDI_SPEC_PATH" "$CDI_SPEC_BACKUP_PATH"
                        set_nvidia_runtime_mode auto || true
                        set +e
                        GPU_TEST_OUTPUT="$(timeout 30 docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_NAME" 2>&1)"
                        GPU_TEST_STATUS=$?
                        set -e

                        if [ $GPU_TEST_STATUS -ne 0 ] && echo "$GPU_TEST_OUTPUT" | grep -q "libnvidia-ml.so.1"; then
                            echo "→ Legacy mode cannot find libnvidia-ml.so.1; retrying after WSL library staging ..."
                            prepare_wsl_legacy_libs || true
                            set +e
                            GPU_TEST_OUTPUT="$(timeout 30 docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_NAME" 2>&1)"
                            GPU_TEST_STATUS=$?
                            set -e
                        fi

                        if [ $GPU_TEST_STATUS -eq 0 ]; then
                            echo "✓ Docker GPU access confirmed in auto/legacy runtime mode."
                            echo "  Keeping ${CDI_SPEC_PATH} disabled at ${CDI_SPEC_BACKUP_PATH}."
                            echo "  If you want to re-enable CDI later:"
                            echo "    1) sudo mv ${CDI_SPEC_BACKUP_PATH} ${CDI_SPEC_PATH}"
                            echo "    2) sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi"
                            echo "    3) sudo systemctl restart docker"
                            GPU_FLAGS="--gpus all"
                        else
                            echo "⚠  Auto/legacy retry also failed; restoring CDI spec and runtime mode."
                            sudo mv "$CDI_SPEC_BACKUP_PATH" "$CDI_SPEC_PATH"
                            set_nvidia_runtime_mode cdi || true
                        fi
                    else
                        echo "⚠  ${CDI_SPEC_PATH} not found; cannot perform CDI-disable retry."
                    fi
                fi
            fi
            echo "   Test output:"
            echo "$GPU_TEST_OUTPUT" | sed 's/^/     /'
            echo "   Common causes: CDI spec missing/stale, NVIDIA Container Toolkit not configured,"
            echo "   or (on Docker Desktop/WSL2) GPU support not enabled in Docker Desktop settings."
            echo "   Docs: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        fi
    else
        echo "⚠  NVIDIA Container Toolkit is not verified — skipping GPU setup and running CPU-only."
        echo "   Install/fix the toolkit and rerun this script:"
        echo "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
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
