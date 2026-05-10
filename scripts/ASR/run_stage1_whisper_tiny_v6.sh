#!/bin/bash
# Stage1 training: Whisper-tiny v6 (per-modality interleave: asr/sound/emotion 0.65/0.25/0.10).
# Same launcher as DAC-VAE v6; only CONFIG path changes.

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

export PATH=$ENV_PREFIX/bin:$PATH

# Caches → /mnt/tmp (DDN at 100%, /mnt/tmp xfs has plenty)
export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# whisper-tiny.en/processor_config.json doesn't exist on HF Hub. Anonymous
# tier returns 401 (rate-limit) and even with HF_TOKEN the workers' subprocess
# env still hits the network and crashes. Force offline mode — the v5 cache at
# /mnt/tmp/cache/huggingface/hub/models--openai--whisper-tiny.en/ has the
# files we actually need (model + feature_extractor) and the .no_exist
# sentinel for the missing processor_config.json. v5 ran 12.6k steps from
# this same cache before crashing on a transient online 401.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA libs from env site-packages
NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

# DeepSpeed JIT-compiles fused_adam; nvcc rejects gcc>11. Force system gcc 9.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

export WANDB_PROJECT=qwen3_5ae-asr
export WANDB_API_KEY=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

cd "$REPO"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[s1-whisper-tiny-v6] launching on $NGPU GPUs"

FORCE_TORCHRUN=1 NPROC_PER_NODE=$NGPU \
    $ENV_PREFIX/bin/llamafactory-cli train configs/ASR/stage1_whisper_tiny_v6.yaml
