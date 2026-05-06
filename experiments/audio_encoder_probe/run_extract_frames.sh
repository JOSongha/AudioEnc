#!/usr/bin/env bash
# Wrapper: env-setup + extract_frames.py
set -e

AUDIOLLM_PREFIX=/mnt/tmp/miniconda3/envs/audiollm
NVIDIA_LIBS=$AUDIOLLM_PREFIX/lib/python3.11/site-packages/nvidia

export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export TMPDIR=/mnt/tmp/cache/tmp
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$TMPDIR"

export LD_LIBRARY_PATH=\
$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:\
$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:$NVIDIA_LIBS/cuda_nvrtc/lib:\
$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:\
$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nvtx/lib:\
$AUDIOLLM_PREFIX/lib:${LD_LIBRARY_PATH:-}

export LD_PRELOAD=$AUDIOLLM_PREFIX/lib/glibc_stub.so
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec conda run -n audiollm --no-capture-output \
    python "$SCRIPT_DIR/extract_frames.py" "$@"
