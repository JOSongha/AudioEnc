#!/bin/bash
# Encodec_24k Stage-1 baseline sweep — same 9 evals × 8 ckpts (28K..35K).
# Each watcher waits for the corresponding v3 eval PID to exit, then launches
# the encodec eval on the freed GPU. GPU 7 (listen_official → asr_external) is
# excluded — too long a queue.

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

CKPT_PARENT=/mnt/tmp/Qwen3.5_encodec_24k_Stage1_jos/Qwen3.5AE-ASR-Stage1-encodec-24k
OUT_ROOT=/mnt/tmp/Qwen3.5_encodec_24k_Stage1_jos/eval_v3_baseline
LOG_ROOT=$OUT_ROOT/_logs
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CKPTS=28000,29000,30000,31000,32000,33000,34000,35000

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

# (v3_pid : gpu : eval_module : extra_args)
# Bump --batch-size for evals that haven't yet been launched (gpu 0/1/3/4/5).
# esc50 (gpu2) + listen_mcqa (gpu6) already started — left at default.
JOBS=(
  "93665:0:eval_librispeech_wer:--split test.clean --batch-size 16"
  "93668:1:eval_clotho_caption:--split validation --batch-size 16"
  "93667:2:eval_esc50_acc:--all"
  "93672:3:eval_audioset_map:--batch-size 16"
  "93674:4:eval_fsd50k_map:--batch-size 16"
  "93675:5:eval_source_emotion:--batch-size 16"
  "93676:6:eval_listen_mcqa:"
)

run_one() {
  local pid="$1" gpu="$2" mod="$3" extra="$4"
  local LOG="$LOG_ROOT/${mod}.log"
  echo "[encodec][gpu$gpu] waiting for v3 PID $pid ($mod) to exit..." | tee -a "$LOG"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  echo "[encodec][gpu$gpu] $mod start $(date +%H:%M:%S)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m "evaluation.stage2.$mod" \
    --ckpt-root "$CKPT_PARENT" \
    --out-root  "$OUT_ROOT/$mod" \
    --ckpts     "$CKPTS" \
    $extra \
    >>"$LOG" 2>&1
  rc=$?
  echo "[encodec][gpu$gpu] $mod exit=$rc $(date +%H:%M:%S)" | tee -a "$LOG"
}

for entry in "${JOBS[@]}"; do
  IFS=: read -r pid gpu mod extra <<< "$entry"
  run_one "$pid" "$gpu" "$mod" "$extra" &
done
wait
echo "[encodec] all 7 evals done $(date +%H:%M:%S)" | tee -a "$LOG_ROOT/_done.log"
