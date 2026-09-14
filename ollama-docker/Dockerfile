FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture and install Ollama binary
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64) ARCH="amd64" ;; \
        aarch64|arm64) ARCH="arm64" ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://ollama.com/download/ollama-linux-${ARCH}.tar.zst" \
        | tar --zstd -x -C /usr && \
    rm -rf /var/lib/apt/lists/*

# Create default models directory
RUN mkdir -p /root/.ollama

# GPU / NVIDIA support
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0

# Expose Ollama's API port
EXPOSE 11434

# Set the entrypoint
ENTRYPOINT ["ollama"]
CMD ["serve"]
