# Default to the NVIDIA official PyTorch image
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE}

# ------------------------------------------------------------------------------
# Build arguments (boolean flags). You can override these at build time:
#   docker build --build-arg INSTALL_FLASHATTN=true -t your_image_name .
# ------------------------------------------------------------------------------
ARG INSTALL_AS_DEV=true
ARG INSTALL_BNB=false
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=true
ARG INSTALL_FLASHATTN=false
ARG INSTALL_LIGER_KERNEL=true
ARG INSTALL_HQQ=false
ARG INSTALL_EETQ=false
ARG INSTALL_MEGATRON=false

# Optional: proxy (if needed). Example:
#   docker build --build-arg PIP_OPTIONS="--proxy http://10.169.41.145:10968/" -t your_image .
ARG PIP_OPTIONS=

# ------------------------------------------------------------------------------
# Set up environment variables
# ------------------------------------------------------------------------------
ENV MAX_JOBS=4
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# ------------------------------------------------------------------------------
# Set the working directory
# ------------------------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------------------------
# Copy and install Python requirements
# ------------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN set -eux; \
    # Upgrade pip first
    python -m pip install ${PIP_OPTIONS} --upgrade pip && \
    # Install from requirements.txt
    python -m pip install ${PIP_OPTIONS} -r requirements.txt

# ------------------------------------------------------------------------------
# Copy the rest of the application code
# (Do this *after* installing dependencies to leverage Docker layer caching)
# ------------------------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------------------------
# Install LLaMA Factory + optional packages, matching shell script logic
# ------------------------------------------------------------------------------
RUN set -eux; \
    EXTRA_PACKAGES="metrics"; \
    if [ "${INSTALL_BNB}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
    fi; \
    if [ "${INSTALL_VLLM}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "${INSTALL_DEEPSPEED}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    if [ "${INSTALL_LIGER_KERNEL}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},liger-kernel"; \
    fi; \
    if [ "${INSTALL_HQQ}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},hqq"; \
    fi; \
    if [ "${INSTALL_EETQ}" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},eetq"; \
    fi; \
    # Decide dev vs non-dev install
    if [ "${INSTALL_AS_DEV}" = "true" ]; then \
        pip install ${PIP_OPTIONS} -e ".[$EXTRA_PACKAGES]"; \
    else \
        pip install ${PIP_OPTIONS} ".[$EXTRA_PACKAGES]"; \
    fi; \
    # Rebuild FlashAttention if requested
    if [ "${INSTALL_FLASHATTN}" = "true" ]; then \
        pip uninstall -y transformer-engine flash-attn; \
        pip uninstall -y ninja; \
        pip install ${PIP_OPTIONS} ninja; \
        pip install ${PIP_OPTIONS} --no-cache-dir flash-attn --no-build-isolation; \
    fi; \
    # Optionally install Megatron
    if [ "${INSTALL_MEGATRON}" = "true" ]; then \
        mkdir -p 3rd_party; \
        cd 3rd_party; \
        git clone https://github.com/NVIDIA/Megatron-LM.git; \
        cd Megatron-LM; \
        if [ "${INSTALL_AS_DEV}" = "true" ]; then \
            pip install ${PIP_OPTIONS} -e .; \
        else \
            pip install ${PIP_OPTIONS} .; \
        fi; \
    fi; \
    # Export local bin path for convenience
    echo "export PATH=\${HOME}/.local/bin:\${PATH}" >> ~/.bashrc

# ------------------------------------------------------------------------------
# (Optional) Create volumes for caching and data/output directories
# ------------------------------------------------------------------------------
VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# ------------------------------------------------------------------------------
# Expose ports commonly used in LLM/Gradio deployments
# ------------------------------------------------------------------------------
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

ENV API_PORT=8000
EXPOSE 8000

# ------------------------------------------------------------------------------
# Set default command (modify if needed)
# ------------------------------------------------------------------------------
CMD ["/bin/bash"]
