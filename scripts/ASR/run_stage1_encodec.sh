#!/bin/bash
# Stage1 training: EnCodec-24k encoder (jos environment).
# Single-node multi-GPU via torchrun + deepspeed z2.
# All cache/output paths on /mnt/tmp (xfs scratch — Lustre quota saturated).

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

# Put env bin ahead of /usr/local/bin so `torchrun` resolves to the audio_lmf
# Python 3.10 (with llamafactory editable-installed), not the system Python 3.8.
export PATH=$ENV_PREFIX/bin:$PATH

# Caches → /mnt/tmp (DDN at 100%, /mnt/tmp xfs has plenty)
export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

# NCCL/torch
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA libs from env site-packages
NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

# DeepSpeed JIT-compiles fused_adam for the optimizer. nvcc rejects gcc > 11,
# but the conda env ships gcc 13. Force the system gcc 9 (supported by nvcc).
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# wandb is online-only — disable to keep runs purely local.
export WANDB_MODE=disabled

cd "$REPO"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[s1-encodec] launching on $NGPU GPUs"

FORCE_TORCHRUN=1 NPROC_PER_NODE=$NGPU \
    $ENV_PREFIX/bin/llamafactory-cli train configs/ASR/stage1_encodec_24k.yaml
