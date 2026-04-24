#!/usr/bin/env bash

# Tested on nvcr.io/nvidia/pytorch:24.02-py3 image.

# Move to repo root directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR/..

# Define default environments
export MAX_JOBS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Define installation arguments
INSTALL_AS_DEV=${INSTALL_AS_DEV:-true}

INSTALL_TORCH=${INSTALL_TORCH:-false}
INSTALL_BNB=${INSTALL_BNB:-false}
INSTALL_VLLM=${INSTALL_VLLM:-false}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-true}
INSTALL_FLASHATTN=${INSTALL_FLASHATTN:-false}
INSTALL_LIGER_KERNEL=${INSTALL_LIGER_KERNEL:-true}
INSTALL_HQQ=${INSTALL_HQQ:-false}
INSTALL_EETQ=${INSTALL_EETQ:-false}
INSTALL_WANDB=${INSTALL_WANDB:-true}
INSTALL_MEGATRON=${INSTALL_MEGATRON:-false}

# For environments that require a proxy workaround, use the following
# option. (Note that the server address is temporary, and may require
# a new server location.)
# PIP_OPTIONS=--proxy=http://10.169.41.145:10968/
PIP_OPTIONS=

# Check pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
  echo "pyproject.toml not found in the current directory. Exiting."
  exit 1
fi

# Configure pip and install from pyproject.toml
python -m pip install ${PIP_OPTIONS} --upgrade pip

# Install LLaMA Factory with optional packages
EXTRA_PACKAGES="metrics"
if [ "$INSTALL_TORCH" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,torch"
fi
if [ "$INSTALL_BNB" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,bitsandbytes"
fi
if [ "$INSTALL_VLLM" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,vllm"
fi
if [ "$INSTALL_DEEPSPEED" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,deepspeed"
fi
if [ "$INSTALL_LIGER_KERNEL" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,liger-kernel"
fi
if [ "$INSTALL_HQQ" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,hqq"
fi
if [ "$INSTALL_EETQ" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,eetq"
fi
if [ "$INSTALL_WANDB" == "true" ]; then
    EXTRA_PACKAGES="$EXTRA_PACKAGES,wandb"
fi

if [ "$INSTALL_AS_DEV" == "true" ]; then
    pip install ${PIP_OPTIONS} -e ".[$EXTRA_PACKAGES]"
else
    pip install ${PIP_OPTIONS} ".[$EXTRA_PACKAGES]"
fi

# Rebuild flash attention if enabled
if [ "$INSTALL_FLASHATTN" == "true" ]; then
    pip uninstall -y transformer-engine flash-attn
    pip uninstall -y ninja
    pip install ${PIP_OPTIONS} ninja
    pip install ${PIP_OPTIONS} --no-cache-dir flash-attn --no-build-isolation
fi

if [ "$INSTALL_MEGATRON" == "true" ]; then
    mkdir -p 3rd_party
    pushd 3rd_party
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM

    if [ "$INSTALL_AS_DEV" == "true" ]; then
        pip install ${PIP_OPTIONS} -e .
    else
        pip install ${PIP_OPTIONS} .
    fi
    popd
fi

echo "export PATH=${HOME}/.local/bin:${PATH}" >> ~/.bashrc
if [ -f "${HOME}/.zshrc" ]; then
    echo "export PATH=${HOME}/.local/bin:${PATH}" >> ~/.zshrc
fi

echo "Setup complete."
