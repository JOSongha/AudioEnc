#!/bin/bash
# Stage 2 LoRA from Phase 1b-A (multi-task projector pretrain) ckpt as base.
# Companion to run_stage2_wavtok.sh (which starts from ASR-only Stage 1 ckpt-100000).
# 8-GPU NSML, deepspeed z2 (no fused), wandb.

source /mnt/tmp/miniconda3/etc/profile.d/conda.sh
conda activate audiollm

# wandb auth check
if [[ "${WANDB_MODE:-online}" != "offline" && "${WANDB_MODE:-online}" != "disabled" ]]; then
    if ! python -c "import wandb; wandb.Api().viewer" 2>/dev/null; then
        echo "[ERROR] wandb 인증 안 됨. 'wandb login' 후 재실행." >&2
        exit 1
    fi
fi

# Environment variables
export WANDB_MODE=online
export WANDB_PROJECT=audiollm-trainer
export WANDB_API_KEY=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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Cache redirects (~ > 30 GB → node hang)
export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

AUDIOLLM_PREFIX=/mnt/tmp/miniconda3/envs/audiollm
NVIDIA_LIBS=$AUDIOLLM_PREFIX/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:$NVIDIA_LIBS/cuda_nvrtc/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nvtx/lib:$AUDIOLLM_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$AUDIOLLM_PREFIX/lib/glibc_stub.so
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# Training
FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 \
    llamafactory-cli train \
    /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/configs/qwen3_5ae-asr/stage2_wavtok_40_unify_from_stage1b.yaml
