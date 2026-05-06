#!/bin/bash
# Overnight GPU filler: per-GPU chain that waits for current eval to clear,
# then runs additional fillers so no GPU sits idle while the long-tail evals
# (audioset_map ~5h, fsd50k_map ~17h) are still going.

set -u
set -o pipefail

REPO=/mnt/ddn/users/jos/audiollm-trainer
ENV_PREFIX=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf
PY=$ENV_PREFIX/bin/python

V3_CKPT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v3
V3_OUT=/mnt/tmp/Qwen3.5_dac_vae_v3_Stage1_jos/eval_v3
ENC_CKPT=/mnt/tmp/Qwen3.5_encodec_24k_Stage1_jos/Qwen3.5AE-ASR-Stage1-encodec-24k
ENC_OUT=/mnt/tmp/Qwen3.5_encodec_24k_Stage1_jos/eval_v3_baseline
LOG_ROOT=$V3_OUT/_logs/fill
mkdir -p "$LOG_ROOT"

V3_CKPTS=10000,20000,30000,40000,50000,60000,70000,80000,90000,94000
V3_DENSE=5000,15000,25000,35000,45000,55000,65000,75000,85000
ENC_CKPTS=28000,29000,30000,31000,32000,33000,34000,35000

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

# Block until GPU has zero compute apps (uses host PID lookup that auto-strips ghost burns).
wait_gpu_idle() {
  local gpu=$1
  while true; do
    n=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | grep -cv '^$')
    [ "$n" -eq 0 ] && break
    sleep 60
  done
}

# run_eval gpu mod ckpt_root out_dir ckpts extra
run_eval() {
  local gpu=$1 mod=$2 ckpt_root=$3 out_dir=$4 ckpts=$5 extra=$6
  local LOG=$LOG_ROOT/gpu${gpu}_${mod}_$(echo "$ckpt_root" | grep -o 'dac_vae_v3\|encodec_24k').log
  mkdir -p "$out_dir"
  echo "[fill][gpu$gpu] $(date +%H:%M:%S) START $mod ckpts=$ckpts" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m "evaluation.stage2.$mod" \
    --ckpt-root "$ckpt_root" --out-root "$out_dir" \
    ${ckpts:+--ckpts $ckpts} $extra \
    >>"$LOG" 2>&1
  echo "[fill][gpu$gpu] $(date +%H:%M:%S) END $mod rc=$?" | tee -a "$LOG"
}

# run_asr_external_loop gpu (asr_external takes single ckpt-root, loop 8 times)
run_asr_external_loop() {
  local gpu=$1 ckpt_parent=$2 out_root=$3 ckpts_list=$4
  for step in $ckpts_list; do
    local CKPT_DIR="$ckpt_parent/checkpoint-$step"
    local OUT_DIR="$out_root/checkpoint-$step"
    [ -f "$OUT_DIR/summary.json" ] && { echo "[fill][gpu$gpu] skip ckpt-$step (done)"; continue; }
    [ ! -d "$CKPT_DIR" ] && { echo "[fill][gpu$gpu] miss ckpt-$step"; continue; }
    mkdir -p "$OUT_DIR"
    local LOG=$LOG_ROOT/gpu${gpu}_asr_external_$(basename $ckpt_parent).log
    wait_gpu_idle "$gpu"
    echo "[fill][gpu$gpu] $(date +%H:%M:%S) START asr_external ckpt-$step" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m evaluation.stage2.eval_asr_external \
      --ckpt-root "$CKPT_DIR" --out-root "$OUT_DIR" \
      --datasets mls voxpopuli gigaspeech \
      --max-samples 500 --batch-size 16 \
      >>"$LOG" 2>&1
    echo "[fill][gpu$gpu] $(date +%H:%M:%S) END asr_external ckpt-$step rc=$?" | tee -a "$LOG"
  done
}

# Initial settle: wait 120s so we don't trip on encodec evals just spawned.
sleep 120
echo "[fill] $(date) starting per-GPU chains"

# ============================================================================
# Per-GPU chains (parallel). Each waits for prior work on its GPU to clear,
# then runs filler queue. Order chosen by current chain depth on each GPU:
#   - GPU 4: only fsd50k_map (~17h v3) → encodec fsd50k → no further filler
#   - GPU 3: only audioset_map (~5h v3) → encodec audioset → encodec asr_external
#   - others get text_retention / encodec listen_official / etc.
# ============================================================================

# GPU 0: v3 libri → encodec libri → v3 text_retention → encodec text_retention
{
  wait_gpu_idle 0
  run_eval 0 eval_text_retention "$V3_CKPT"  "$V3_OUT/eval_text_retention"  "$V3_CKPTS"  "--batch-size 16"
  wait_gpu_idle 0
  run_eval 0 eval_text_retention "$ENC_CKPT" "$ENC_OUT/eval_text_retention" "$ENC_CKPTS" "--batch-size 16"
} >>"$LOG_ROOT/_chain_gpu0.log" 2>&1 &

# GPU 1: v3 clotho → encodec clotho → encodec asr_external (1 of 2 instances)
{
  wait_gpu_idle 1
  run_asr_external_loop 1 "$ENC_CKPT" "$ENC_OUT/eval_asr_external" "28000 29000 30000 31000"
} >>"$LOG_ROOT/_chain_gpu1.log" 2>&1 &

# GPU 2: encodec esc50 → encodec listen_official (was excluded from baseline)
{
  wait_gpu_idle 2
  run_eval 2 eval_listen_official "$ENC_CKPT" "$ENC_OUT/eval_listen_official" "$ENC_CKPTS" "--batch-size 16"
} >>"$LOG_ROOT/_chain_gpu2.log" 2>&1 &

# GPU 3: v3 audioset (long) → encodec audioset → no filler (already long enough)
# GPU 4: v3 fsd50k (very long) → encodec fsd50k → no filler
# (skipping these — encodec_baseline already chained, no idle time)

# GPU 5: v3 source_emotion → encodec source_emotion → encodec asr_external (2 of 2)
{
  wait_gpu_idle 5
  run_asr_external_loop 5 "$ENC_CKPT" "$ENC_OUT/eval_asr_external" "32000 33000 34000 35000"
} >>"$LOG_ROOT/_chain_gpu5.log" 2>&1 &

# GPU 6: encodec listen_mcqa → v3 listen_mcqa dense (5K offset)
{
  wait_gpu_idle 6
  run_eval 6 eval_listen_mcqa "$V3_CKPT" "$V3_OUT/eval_listen_mcqa_dense" "$V3_DENSE" "--batch-size 16"
} >>"$LOG_ROOT/_chain_gpu6.log" 2>&1 &

# GPU 7: v3 listen_official → asr_external 10 ckpts (already queued by asr_ext_gpu7.sh)
#       → encodec listen_official → v3 listen_official dense
{
  wait_gpu_idle 7
  run_eval 7 eval_listen_official "$ENC_CKPT" "$ENC_OUT/eval_listen_official" "$ENC_CKPTS" "--batch-size 16"
  wait_gpu_idle 7
  run_eval 7 eval_listen_official "$V3_CKPT" "$V3_OUT/eval_listen_official_dense" "$V3_DENSE" "--batch-size 16"
} >>"$LOG_ROOT/_chain_gpu7.log" 2>&1 &

wait
echo "[fill] $(date) all chains done"
