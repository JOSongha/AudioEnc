#!/usr/bin/env bash
# Run extraction for all 5 families sequentially.
# Resume-safe: skips already-completed utterances.
# Logs per-family to /mnt/tmp/cache/extract_{family}.log

set -e

AUDIOLLM_PREFIX=/mnt/tmp/miniconda3/envs/audiollm
NVIDIA_LIBS=$AUDIOLLM_PREFIX/lib/python3.11/site-packages/nvidia

export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export WANDB_DIR=/mnt/tmp/cache/wandb
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

export LD_LIBRARY_PATH=\
$NVIDIA_LIBS/nccl/lib:\
$NVIDIA_LIBS/cudnn/lib:\
$NVIDIA_LIBS/cublas/lib:\
$NVIDIA_LIBS/cuda_runtime/lib:\
$NVIDIA_LIBS/cuda_cupti/lib:\
$NVIDIA_LIBS/cuda_nvrtc/lib:\
$NVIDIA_LIBS/cufft/lib:\
$NVIDIA_LIBS/curand/lib:\
$NVIDIA_LIBS/cusolver/lib:\
$NVIDIA_LIBS/cusparse/lib:\
$NVIDIA_LIBS/nvjitlink/lib:\
$NVIDIA_LIBS/nvtx/lib:\
$AUDIOLLM_PREFIX/lib:\
${LD_LIBRARY_PATH:-}

export LD_PRELOAD=$AUDIOLLM_PREFIX/lib/glibc_stub.so
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST=/mnt/tmp/cache/cmu_arctic_7/manifest.csv
OUT_DIR="$SCRIPT_DIR/cmu_arctic_7"
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

run_family() {
    local family=$1
    local logfile=/mnt/tmp/cache/extract_${family}.log
    echo "========================================" | tee -a "$logfile"
    echo "START: $family  $(date)" | tee -a "$logfile"
    echo "========================================" | tee -a "$logfile"

    conda run -n audiollm --no-capture-output \
        python "$SCRIPT_DIR/extract.py" \
            --family "$family" \
            --manifest "$MANIFEST" \
            --out_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
        2>&1 | tee -a "$logfile"

    echo "DONE: $family  $(date)" | tee -a "$logfile"
}

run_family whisper_tiny
run_family whisper_small
run_family wavtok_40_unify
run_family dacvae_stage1
run_family dacvae_stage2

echo ""
echo "ALL FAMILIES COMPLETE: $(date)"
echo "Output: $OUT_DIR"
