#!/usr/bin/env bash
# Single-node 8-GPU launcher for Stage-2 whisper-tiny (v2 manifest, emoFull/asr033/env05/txt03).
# Mirrors run_whisper_small_8gpu.sh but points at stage2_whisper_tiny.yaml.
set -eo pipefail

source /mnt/ddn/users/jos/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
  || source /mnt/ddn/users/jos/miniforge3/bin/activate
conda activate audio_lmf

ENV_LIB=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib
export LD_PRELOAD="${ENV_LIB}/libstdc++.so.6:${ENV_LIB}/glibc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3_5ae-asr}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-/mnt/ddn/users/jos/audiollm-trainer/configs/qwen3_5ae-asr/stage2_whisper_tiny.yaml}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29503}"

echo "[whisper-tiny launch] config: $CONFIG"
echo "[whisper-tiny launch] nproc:  $NPROC"

cd /mnt/ddn/users/jos/audiollm-trainer

FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC}" \
llamafactory-cli train "${CONFIG}"
