#!/bin/bash
# Autonomous overnight dispatcher for v3 eval gap-fill.
# Polls all 8 GPUs every 5 min; when one is idle, pops next job from queue
# and dispatches. Runs in background, no Claude wakeup needed.
#
# Format of queue file lines: mod|out_subdir|ckpts_csv|extra
#   - mod: module name (e.g. eval_fsd50k_map) or sentinel "WER" for whisper script
#   - out_subdir: subdir under $V3_OUT to write results to
#   - ckpts_csv: comma-separated ckpts for this job
#   - extra: extra CLI args (e.g. "--batch-size 16" or "--all --batch-size 16")
# Lines starting with # or empty are skipped.

set -u

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

V3_CKPT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v3
V3_OUT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3

LOG_ROOT=$V3_OUT/_logs/dispatch
QUEUE=$LOG_ROOT/queue_pending.txt
MAIN_LOG=$LOG_ROOT/_auto_dispatcher.log
PID_FILE=$LOG_ROOT/_auto_dispatcher.pid

mkdir -p "$LOG_ROOT"

# Singleton: refuse to start if another instance running
if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
  echo "[auto-disp] already running pid=$(cat $PID_FILE)" | tee -a "$MAIN_LOG"
  exit 1
fi
echo $$ > "$PID_FILE"

cd "$REPO"
export PATH=$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/tmp/cache/huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONIOENCODING=utf-8

NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

log() { echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$MAIN_LOG" >/dev/null; }

# Memory-based idle: <1500 MiB = idle (5x phantom-margin since real eval needs ~15-80GB)
gpu_idle() {
  local g=$1 mem
  mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$mem" -lt 1500 ]
}

dispatch_job() {
  local g=$1 mod=$2 out_sub=$3 ckpts=$4 extra=$5
  local out_dir="$V3_OUT/$out_sub"
  mkdir -p "$out_dir"
  local jid=$(date +%H%M%S)_g${g}_${mod}
  local LOG=$LOG_ROOT/auto_${jid}.log

  log "[gpu$g] launch mod=$mod sub=$out_sub ckpts=$ckpts"

  if [ "$mod" = "WER" ]; then
    # WER whisper sits at evaluation/eval_testclean_wer_whisper.py (NOT under stage2)
    CUDA_VISIBLE_DEVICES=$g nohup $PY evaluation/eval_testclean_wer_whisper.py \
      --ckpt-root "$V3_CKPT" --out-root "$out_dir" \
      --ckpts $ckpts $extra \
      > "$LOG" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$g nohup $PY -m "evaluation.stage2.$mod" \
      --ckpt-root "$V3_CKPT" --out-root "$out_dir" \
      --ckpts $ckpts $extra \
      > "$LOG" 2>&1 &
  fi
  log "[gpu$g] pid=$! log=$LOG"
}

pop_job() {
  # Atomically pop first non-comment, non-empty line
  awk 'NF && $0 !~ /^[[:space:]]*#/ {print; exit}' "$QUEUE"
}

remove_job_line() {
  local job=$1
  awk -v j="$job" 'BEGIN{popped=0} {
    if (!popped && $0 == j) { popped=1; next }
    print
  }' "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
}

log "[start] auto dispatcher pid=$$ queue=$QUEUE"

while true; do
  # Skip if queue empty AND all GPUs done
  if ! pop_job > /dev/null 2>&1 || [ -z "$(pop_job)" ]; then
    busy=0
    for g in 0 1 2 3 4 5 6 7; do gpu_idle "$g" || busy=$((busy+1)); done
    if [ "$busy" -eq 0 ]; then
      log "[done] queue empty + all GPUs idle — exiting"
      rm -f "$PID_FILE"
      exit 0
    fi
    sleep 60
    continue
  fi

  for g in 0 1 2 3 4 5 6 7; do
    gpu_idle "$g" || continue
    # Re-confirm idle after 10s (avoid race with other launches)
    sleep 10
    gpu_idle "$g" || continue

    job=$(pop_job)
    [ -z "$job" ] && break
    remove_job_line "$job"

    IFS='|' read -r mod out_sub ckpts extra <<< "$job"
    if [ -z "$mod" ] || [ -z "$out_sub" ] || [ -z "$ckpts" ]; then
      log "[skip] malformed job: $job"
      continue
    fi
    dispatch_job "$g" "$mod" "$out_sub" "$ckpts" "$extra"

    # Let the new process bind GPU before re-polling
    sleep 60
  done

  sleep 60
done
