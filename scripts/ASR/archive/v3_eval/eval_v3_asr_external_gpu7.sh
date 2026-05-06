#!/bin/bash
# Watcher: wait for listen_official (PID $LISTEN_OFFICIAL_PID) on GPU 7 to exit,
# then run eval_asr_external on GPU 7 for the 10 sampled v3 checkpoints.
# eval_asr_external expects --ckpt-root as a single ckpt dir (no --ckpts loop),
# so we call it once per checkpoint.

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

CKPT_PARENT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v3
BASE_MODEL=/mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B
OUT_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3/eval_asr_external
LOG_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3/_logs
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CKPTS=(10000 20000 30000 40000 50000 60000 70000 80000 90000 94000)
LISTEN_OFFICIAL_PID="${1:-93677}"

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

LOG=$LOG_ROOT/eval_asr_external.log
echo "[asr-ext-watcher] waiting for listen_official PID=$LISTEN_OFFICIAL_PID to exit..." | tee -a "$LOG"
while kill -0 "$LISTEN_OFFICIAL_PID" 2>/dev/null; do sleep 30; done
echo "[asr-ext-watcher] GPU 7 free → starting asr_external sweep" | tee -a "$LOG"

for step in "${CKPTS[@]}"; do
  CKPT_DIR="$CKPT_PARENT/checkpoint-$step"
  OUT_DIR="$OUT_ROOT/checkpoint-$step"
  if [ -f "$OUT_DIR/summary.json" ]; then
    echo "[asr-ext] skip checkpoint-$step (summary exists)" | tee -a "$LOG"
    continue
  fi
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[asr-ext] checkpoint-$step missing — skip" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "[asr-ext] checkpoint-$step start $(date +%H:%M:%S)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=7 $PY -m evaluation.stage2.eval_asr_external \
    --ckpt-root "$CKPT_DIR" \
    --base-model "$BASE_MODEL" \
    --out-root  "$OUT_DIR" \
    --datasets mls voxpopuli gigaspeech \
    --max-samples 500 --batch-size 8 \
    >>"$LOG" 2>&1
  rc=$?
  echo "[asr-ext] checkpoint-$step exit=$rc $(date +%H:%M:%S)" | tee -a "$LOG"
done

echo "[asr-ext-watcher] done $(date +%H:%M:%S)" | tee -a "$LOG"
