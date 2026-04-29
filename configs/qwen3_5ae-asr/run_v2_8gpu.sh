#!/usr/bin/env bash
# Single-node 8-GPU launcher for Stage 2 v2 (epoch-random manifest).
# Differs from run_nsml.sh in that this is for a single-node local run, not
# the multi-node NSML scheduler. Hardcodes nproc_per_node=8.
#
# Pre-flight: caller is expected to confirm the v2 manifest exists at
# /mnt/tmp/listen_analysis/train_manifest/stage2_combined_shards_eprandom/
# (built by scripts/emo/build_epoch_random_manifest.py) and that no other
# heavy GPU process is running (eval / leftover Stage 2 v1 worker).
set -eo pipefail

# ---- env ----
source /mnt/ddn/users/jos/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
  || source /mnt/ddn/users/jos/miniforge3/bin/activate
conda activate audio_lmf

ENV_LIB=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib
export LD_PRELOAD="${ENV_LIB}/libstdc++.so.6:${ENV_LIB}/glibc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3_5ae-asr}"
# WANDB_API_KEY assumed already exported in caller's shell

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-/mnt/ddn/users/jos/audiollm-trainer/configs/qwen3_5ae-asr/stage2_v2.yaml}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "[v2 launch] config: $CONFIG"
echo "[v2 launch] nproc:  $NPROC"

cd /mnt/ddn/users/jos/audiollm-trainer

FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC}" \
llamafactory-cli train "${CONFIG}"
