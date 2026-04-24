#!/usr/bin/env bash
# NSML entrypoint for Qwen3.5AE ASR Stage1 (projector-only) training
set -eo pipefail

# ---- activate conda env (audio_lmf = py3.10 + llamafactory-compatible pins) ----
source /mnt/fr20tb/wbl_residency/jos/ddn/miniforge3/bin/activate audio_lmf

# glibc 2.31 host but flash_attn .so needs __libc_single_threaded (glibc 2.32+) — shim via LD_PRELOAD
export LD_PRELOAD="/mnt/fr20tb/wbl_residency/jos/ddn/miniforge3/envs/audio_lmf/lib/glibc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"

# ---- wandb ----
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3_5ae-asr}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ---- NCCL ----
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# ---- CUDA memory (avoid OOM from fragmentation on long-sample batches) ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- suppress fork-after-tokenizer warning spam ----
export TOKENIZERS_PARALLELISM=false

# ---- config path ----
CONFIG="${CONFIG:-/mnt/fr20tb/wbl_residency/jos/ddn/audiollm-trainer/configs/qwen3_5ae-asr/stage1_projector.yaml}"

FORCE_TORCHRUN=1 \
NNODES="${NSML_WORLD_SIZE}" \
NODE_RANK="${NSML_RANK}" \
MASTER_ADDR="${NSML_HOST_RANK0}" \
MASTER_PORT=21267 \
llamafactory-cli train "${CONFIG}"
