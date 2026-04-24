#!/bin/bash
#SBATCH --job-name=audiollm-speechx
#SBATCH --partition=batch-exp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%j-%x-sbatch.out
#SBATCH --error=logs/%j-%x-sbatch.err

################################################################################
# Usage:
#   WANDB_API_KEY=<KEY> [HTTP_PROXY=<proxy>] [HTTPS_PROXY=<proxy>] sbatch \
#       audiollm-speechx.sbatch <TRAIN_YAML>
#
# Where:
#   <TRAIN_YAML> is a path to any llamafactory-cli-compatible YAML file.
#
# This script stores the entire bind.sh in a variable and writes it to each node
# before launching the container. No external bind.sh file is needed.
################################################################################

set -euo pipefail

# Ensure at least one argument: the YAML path
if [ $# -lt 1 ]; then
    echo "Usage: WANDB_API_KEY=<KEY> sbatch $0 <TRAIN_YAML>"
    exit 1
fi

TRAIN_YAML="$1"

# ------------------------------------------------------------------------------
# Check for WANDB_API_KEY
# ------------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set."
    exit 1
fi

# ------------------------------------------------------------------------------
# Environment exports
# ------------------------------------------------------------------------------
export WANDB_MODE=online
export WANDB_PROJECT="Speech X"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# If RUN_NAME is not set, define a default and warn the user
if [ -z "${RUN_NAME:-}" ]; then
  echo "[WARNING] RUN_NAME not specified; defaulting to speechx-${SLURM_NNODES}-$(date +'%Y%m%d-%H%M%S')"
  export RUN_NAME="speechx-${SLURM_NNODES}-$(date +'%Y%m%d-%H%M%S')"
fi

export HF_HOME="/mnt/clovanap/audiollm/datasets/.cache"
export HF_DATASETS_CACHE="/mnt/clovanap/audiollm/datasets/.cache"

# Slurm-based environment variables for distributed training
export NNODES="${SLURM_NNODES}"
export NODE_RANK="${SLURM_NODEID}"
export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)"
export MASTER_PORT=29500
export FORCE_TORCHRUN=1

# ------------------------------------------------------------------------------
# Log CPU info for reference
# ------------------------------------------------------------------------------
lscpu_out="$(lscpu)"

get_lscpu_value() {
    local key="$1"
    echo "${lscpu_out}" | awk -F: -v k="$key" '($1 == k) {gsub(/ /, "", $2); print $2; exit}'
}

num_sockets="$( get_lscpu_value "Socket(s)" )"
num_nodes="$( get_lscpu_value "NUMA node(s)" )"
cores_per_socket="$( get_lscpu_value "Core(s) per socket" )"
cores_per_node=$(( (num_sockets * cores_per_socket) / num_nodes ))

cat <<EOM
[INFO] Slurm job node info:
       NNODES=${NNODES}
       NODE_RANK=${NODE_RANK}
       MASTER_ADDR=${MASTER_ADDR}

[INFO] CPU topology (lscpu):
       num_sockets=${num_sockets}
       num_nodes=${num_nodes}
       cores_per_socket=${cores_per_socket}
       cores_per_node=${cores_per_node}

[INFO] Launching llamafactory-cli train with YAML: ${TRAIN_YAML}
EOM

# ------------------------------------------------------------------------------
# Decide which variables to export into srun
# ------------------------------------------------------------------------------
SRUN_EXPORT="ALL"
if [ -n "${HTTP_PROXY:-}" ] || [ -n "${HTTPS_PROXY:-}" ]; then
    echo "[INFO] HTTP_PROXY or HTTPS_PROXY is set, applying to worker nodes."
    SRUN_EXPORT="ALL,HTTP_PROXY,HTTPS_PROXY"
fi

# ------------------------------------------------------------------------------
# Container image info
# ------------------------------------------------------------------------------
IMAGE_PATH="/mnt/fr20tb/audiollm/images/audiollm-speechx-v1.sqsh"
CONTAINER_NAME="audiollm-speechx-${SLURM_JOB_ID}"

# ------------------------------------------------------------------------------
# Actually run srun, installing your package and then calling run.sh
# ------------------------------------------------------------------------------
srun \
  --kill-on-bad-exit \
  --container-image="${IMAGE_PATH}" \
  --container-name="${CONTAINER_NAME}" \
  --container-mounts="/mnt:/mnt,/raid/scratch:/raid/scratch,/opt/hpc-automation:/opt/hpc-automation" \
  --container-workdir="/mnt/fr20tb/audiollm/codes/python/audiollm-speechx" \
  --export="${SRUN_EXPORT}" \
  --output="logs/%j-%x-%N-srun.out" \
  --error="logs/%j-%x-%N-srun.err" \
  spd2/bind.sh --cpu=off --mem=off --ib=single --cluster=selene \
  /bin/bash -c "pip install -e . && spd2/run.sh \"\${TRAIN_YAML:-configs/speechx-v6/speechx-v6-s2-spd2.yaml}\""

echo "[INFO] Completed Slurm script. Check logs for details."
