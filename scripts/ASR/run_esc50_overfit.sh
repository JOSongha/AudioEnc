#!/bin/bash
# ESC-50 closed-set overfit experiment driver.
#
# Compares how fast each acoustic encoder family lets the full ALM
# (projector + LoRA-LLM) memorize ESC-50 (50-class env-sound, 2000 samples).
# Encoder is frozen throughout — we measure projector + LLM extraction
# capacity given the same upstream features.
#
# Usage:
#   bash scripts/ASR/run_esc50_overfit.sh wavtok
#   bash scripts/ASR/run_esc50_overfit.sh whisper_tiny
#   bash scripts/ASR/run_esc50_overfit.sh whisper_small
#   bash scripts/ASR/run_esc50_overfit.sh dacvae

set -euo pipefail

ENCODER="${1:-}"
case "$ENCODER" in
    wavtok|whisper_tiny|whisper_small|dacvae) ;;
    *)
        echo "usage: $0 {wavtok|whisper_tiny|whisper_small|dacvae}" >&2
        exit 2
        ;;
esac

CFG="/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/configs/qwen3_5ae-asr/esc50_overfit_${ENCODER}.yaml"

source /mnt/tmp/miniconda3/etc/profile.d/conda.sh
conda activate audiollm

# ── wandb auth check ──────────────────────────────────────────────────────────
if [[ "${WANDB_MODE:-online}" != "offline" && "${WANDB_MODE:-online}" != "disabled" ]]; then
    if ! python -c "import wandb; wandb.Api().viewer" 2>/dev/null; then
        echo "[ERROR] wandb 인증이 안 되어 있습니다." >&2
        exit 1
    fi
fi

# ── Environment variables ──────────────────────────────────────────────────────
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

# Caches off ~ (machine dies if home > 30 GB).
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

# ── Training ──────────────────────────────────────────────────────────────────
FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 \
    llamafactory-cli train "$CFG"
