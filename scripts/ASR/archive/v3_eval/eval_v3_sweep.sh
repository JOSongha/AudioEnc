#!/bin/bash
# Sweep all stage2 evals across selected v3 Stage-1 checkpoints.
# - 9 evals × 10 ckpts (interval 10K + final 94K)
# - 1 eval per GPU (single-GPU eval scripts), 8 GPUs concurrent
# - Per-eval iterates ckpts internally via --ckpts comma-list
# - CPU dataloader workers stay default (~12) → 8×12 ~ 96 < 128 cores ✓

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

CKPT_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v3
BASE_MODEL=/mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B
OUT_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3
LOG_ROOT=$OUT_ROOT/_logs
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CKPTS=10000,20000,30000,40000,50000,60000,70000,80000,90000,94000

# Eval scripts (module path : extra args). One eval per GPU.
EVALS=(
  "evaluation.stage2.eval_librispeech_wer:--split test.clean"
  "evaluation.stage2.eval_clotho_caption:--split validation"
  "evaluation.stage2.eval_esc50_acc:--all"
  "evaluation.stage2.eval_audioset_map:"
  "evaluation.stage2.eval_fsd50k_map:"
  "evaluation.stage2.eval_source_emotion:"
  "evaluation.stage2.eval_listen_mcqa:"
  "evaluation.stage2.eval_listen_official:"
  "evaluation.stage2.eval_text_retention:"
)

cd "$REPO"
export PATH=$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/tmp/cache/huggingface

# CUDA libs from env site-packages + libstdc++ shim (required by causal_conv1d)
NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

# CPU cap per process (8 evals × 4 = 32 threads + dataloader workers ~12 each = ~96 < 128 cores)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_EVALS=${#EVALS[@]}
echo "[eval-sweep] $N_EVALS evals × ${CKPTS//,/ } across $NGPU GPUs"
echo "[eval-sweep] out_root=$OUT_ROOT"

# Spawn each eval bound to its own GPU. If N_EVALS > NGPU, the surplus
# evals wait on a FIFO semaphore (8-slot).
SEMAPHORE=$(mktemp -u)
mkfifo "$SEMAPHORE"
exec 3<>"$SEMAPHORE"
rm -f "$SEMAPHORE"
for ((i=0; i<NGPU; i++)); do echo "$i" >&3; done

for entry in "${EVALS[@]}"; do
  EVAL_MOD="${entry%%:*}"
  EXTRA="${entry#*:}"
  EVAL_NAME="${EVAL_MOD##*.}"
  read -u 3 GPU_ID
  (
    LOG=$LOG_ROOT/${EVAL_NAME}.log
    echo "[eval-sweep][gpu$GPU_ID] launch $EVAL_NAME → $LOG" >&2
    CUDA_VISIBLE_DEVICES=$GPU_ID $PY -m "$EVAL_MOD" \
      --ckpt-root "$CKPT_ROOT" \
      --base-model "$BASE_MODEL" \
      --out-root  "$OUT_ROOT/$EVAL_NAME" \
      --ckpts     "$CKPTS" \
      $EXTRA \
      >"$LOG" 2>&1
    rc=$?
    echo "[eval-sweep][gpu$GPU_ID] $EVAL_NAME exit=$rc" >&2
    echo "$GPU_ID" >&3
  ) &
done

wait
echo "[eval-sweep] all done"

# Aggregate at the end
echo "[eval-sweep] aggregating results..."
$PY -m evaluation.stage2.aggregate_results \
  --root "$OUT_ROOT" \
  --out  "$OUT_ROOT/_summary" 2>&1 | tee "$LOG_ROOT/aggregate.log"

echo "[eval-sweep] DONE — see $OUT_ROOT/_summary/"
