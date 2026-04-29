#!/bin/bash
# Stage1 resume: WavTokenizer SEANetEncoder from checkpoint-45000.
#
# Usage:
#   bash scripts/ASR/run_stage1_wavtok_resume.sh

source /mnt/tmp/miniconda3/etc/profile.d/conda.sh
conda activate audiollm

# ── wandb auth check (real API ping; skip if WANDB_MODE=offline/disabled) ────
if [[ "${WANDB_MODE:-online}" != "offline" && "${WANDB_MODE:-online}" != "disabled" ]]; then
    if ! python -c "import wandb; wandb.Api().viewer" 2>/dev/null; then
        echo "[ERROR] wandb 인증이 안 되어 있습니다. 'wandb login' 실행 후 다시 시도하세요." >&2
        exit 1
    fi
fi

# ── Environment variables ──────────────────────────────────────────────────────
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Redirect caches off ~ (machine dies if home > 30GB).
# Triton / CUDA JIT do NOT honor XDG_CACHE_HOME, so set them explicitly.
export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

AUDIOLLM_PREFIX=/mnt/tmp/miniconda3/envs/audiollm
NVIDIA_LIBS=$AUDIOLLM_PREFIX/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:$NVIDIA_LIBS/cuda_nvrtc/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nvtx/lib:$AUDIOLLM_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$AUDIOLLM_PREFIX/lib/glibc_stub.so
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# ── Training configuration ─────────────────
FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 \
    llamafactory-cli train /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/configs/ASR/stage1_wavtok_40_unify_resume.yaml
