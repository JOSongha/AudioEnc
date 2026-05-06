#!/bin/bash
# Wait for the first eval_v3 sweep to finish, then launch the 3 evals that
# failed on first try (clotho_caption arg + 2 GLIBCXX issues, all fixed in
# the patched eval_v3_sweep.sh). Run with OMP caps to avoid CPU overload.

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

CKPT_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v3
BASE_MODEL=/mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B
OUT_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3
LOG_ROOT=$OUT_ROOT/_logs

CKPTS=10000,20000,30000,40000,50000,60000,70000,80000,90000,94000

cd "$REPO"
export PATH=$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/tmp/cache/huggingface

NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Wait until first sweep is done (≤ 1 eval still running)
echo "[resume-failed] waiting for first sweep to wind down..."
while true; do
  n=$(ps -eo cmd 2>/dev/null | grep -c 'evaluation\.stage2\.eval' || true)
  if [ "$n" -le 2 ]; then
    echo "[resume-failed] only $n evals running — proceed"
    break
  fi
  sleep 60
done

# Failed evals (3): re-launch with OMP caps, distinct GPUs, in parallel
RUN() {
  local gpu=$1; local mod=$2; local extra=$3
  local name="${mod##*.}"
  local LOG=$LOG_ROOT/${name}.retry.log
  echo "[resume-failed][gpu$gpu] $name → $LOG"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m "$mod" \
    --ckpt-root "$CKPT_ROOT" \
    --base-model "$BASE_MODEL" \
    --out-root  "$OUT_ROOT/$name" \
    --ckpts     "$CKPTS" \
    $extra \
    >"$LOG" 2>&1
  echo "[resume-failed][gpu$gpu] $name exit=$?"
}

RUN 0 "evaluation.stage2.eval_clotho_caption"  "--split validation" &
RUN 1 "evaluation.stage2.eval_listen_mcqa"     "" &
RUN 2 "evaluation.stage2.eval_text_retention"  "" &
wait

echo "[resume-failed] done"

# Re-aggregate
$PY -m evaluation.stage2.aggregate_results \
  --root "$OUT_ROOT" \
  --out  "$OUT_ROOT/_summary" 2>&1 | tee "$LOG_ROOT/aggregate_after_retry.log"
