#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Usage: run.sh [CONFIG_FILE]
# Runs the LLaMA Factory launcher with torchrun, defaulting to the SPD2 config.
################################################################################

usage() {
  echo "Usage: $0 [CONFIG_FILE]"
  echo "If not specified, defaults to configs/speechx-v6/speechx-v6-s2-spd2.yaml."
  exit 1
}

# Check for help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

# Accept config file path or use default
CONFIG_FILE="${1:-configs/speechx-v6/speechx-v6-s2-spd2.yaml}"
echo "[INFO] Using config file: ${CONFIG_FILE}"

# Torchrun arguments, referencing SLURM environment variables
# Modify nproc_per_node, master_port, etc., as needed
torchrun \
  --nnodes="${SLURM_STEP_NUM_NODES:-1}" \
  --node_rank="${SLURM_NODEID:-0}" \
  --nproc_per_node=8 \
  --master_addr="${HOSTNAME:-127.0.0.1}" \
  --master_port=29500 \
  src/llamafactory/launcher.py \
  "${CONFIG_FILE}"
