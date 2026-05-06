#!/bin/bash
# Unified GPU dispatcher: poll all 8 GPUs; whenever one is idle, pop next job
# from queue.txt and dispatch. Coexists safely with existing per-GPU chain
# watchers — both use "GPU idle" as trigger so no double-launch.

set -u
REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

LOG_ROOT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3/_logs/dispatch
QUEUE=$LOG_ROOT/queue.txt
mkdir -p "$LOG_ROOT"
touch "$QUEUE"

cd "$REPO"
export PATH=$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/tmp/cache/huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

NVIDIA_LIBS=$ENV_PREFIX/lib/python*/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NVIDIA_LIBS/nccl/lib $NVIDIA_LIBS/cudnn/lib $NVIDIA_LIBS/cublas/lib $NVIDIA_LIBS/cuda_runtime/lib $NVIDIA_LIBS/cuda_cupti/lib $NVIDIA_LIBS/cuda_nvrtc/lib $NVIDIA_LIBS/cufft/lib $NVIDIA_LIBS/curand/lib $NVIDIA_LIBS/cusolver/lib $NVIDIA_LIBS/cusparse/lib $NVIDIA_LIBS/nvjitlink/lib $NVIDIA_LIBS/nvtx/lib 2>/dev/null | paste -sd: -):$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$ENV_PREFIX/lib/libstdc++.so.6:$ENV_PREFIX/lib/glibc_compat.so

# GPU-busy check: phantom CUDA contexts (dead python with leaked alloc) report
# as compute apps but use ~8 GB and zero util. A real eval needs >=15 GB. Use
# memory-based threshold so phantom PIDs don't block dispatch forever.
gpu_busy() {
  local g=$1
  local mem
  mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$mem" -ge 15000 ]
}

while true; do
  # If queue is empty + all GPUs idle → exit
  if [ ! -s "$QUEUE" ]; then
    busy=0
    for g in 0 1 2 3 4 5 6 7; do gpu_busy "$g" && busy=$((busy+1)); done
    if [ "$busy" -eq 0 ]; then
      echo "[disp] $(date) queue empty + all GPUs idle — done" | tee -a "$LOG_ROOT/_main.log"
      break
    fi
    sleep 90
    continue
  fi

  for g in 0 1 2 3 4 5 6 7; do
    gpu_busy "$g" && continue
    # Re-confirm idle after 5s (race avoidance with chain watchers)
    sleep 5
    gpu_busy "$g" && continue

    # Atomically pop first non-comment, non-empty job line from queue.
    # Use awk to: emit (skip) header lines, capture first job line, then
    # emit (keep) all remaining lines verbatim. Matched job goes to stdout
    # via a separate file write.
    job=""
    if [ -s "$QUEUE" ]; then
      job=$(awk 'NF && $0 !~ /^#/ {print; exit}' "$QUEUE")
      if [ -n "$job" ]; then
        awk -v j="$job" 'BEGIN{popped=0} {
          if (!popped && $0 == j) { popped=1; next }
          print
        }' "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
      fi
    fi
    [ -z "$job" ] && continue

    # Format: jid|mod|ckpt_root|out_dir|ckpts_csv|extra
    IFS='|' read -r jid mod ckpt out ckpts extra <<< "$job"
    [ -z "$jid" ] || [ -z "$mod" ] || [ -z "$ckpt" ] || [ -z "$out" ] && continue
    mkdir -p "$out"

    LOG=$LOG_ROOT/${jid}.log
    echo "[disp][gpu$g] $(date +%H:%M:%S) launch jid=$jid mod=$mod" | tee -a "$LOG_ROOT/_main.log" "$LOG" >/dev/null

    CUDA_VISIBLE_DEVICES=$g $PY -m "evaluation.stage2.$mod" \
      --ckpt-root "$ckpt" --out-root "$out" \
      ${ckpts:+--ckpts $ckpts} $extra \
      >>"$LOG" 2>&1 &

    sleep 90  # let model bind GPU before next poll iteration
    break
  done

  sleep 30
done
