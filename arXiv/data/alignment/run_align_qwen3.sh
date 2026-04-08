#!/usr/bin/env bash
# Qwen3-ForcedAligner 다중 GPU 런처
# 사용법: bash run_align_qwen3.sh [dataset] [split] [batch_size] [workers_per_gpu]
#   dataset:          mls | gigaspeech | voxpopuli  (기본: mls)
#   split:            train | validation | test      (기본: train)
#   batch_size:       한 번에 처리할 발화 수          (기본: 16)
#   workers_per_gpu:  GPU당 워커 수                  (기본: 2)

set -e

DATASET="${1:-mls}"
SPLIT="${2:-train}"
BATCH="${3:-16}"
WORKERS_PER_GPU="${4:-5}"

NUM_GPUS=8
NUM_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
CACHE_DIR="/mnt/tmp/cache"
OUT_DIR="/mnt/tmp/cache/word_alignments_qwen3"
LOG_DIR="$OUT_DIR/logs"
VENV="/mnt/fr20tb/wbl_residency/jos/.venv310/bin/python3"
SCRIPT="$(dirname "$0")/align_qwen3.py"

mkdir -p "$LOG_DIR"

echo "=== Qwen3 Alignment: $DATASET/$SPLIT (batch=$BATCH, GPUs=$NUM_GPUS, workers/GPU=$WORKERS_PER_GPU, total=$NUM_WORKERS) ==="

pids=()
for rank in $(seq 0 $((NUM_WORKERS - 1))); do
    gpu=$((rank % NUM_GPUS))
    log="$LOG_DIR/${DATASET}_${SPLIT}_rank${rank}.log"
    $VENV "$SCRIPT" \
        --dataset   "$DATASET" \
        --split     "$SPLIT" \
        --gpu       "$gpu" \
        --rank      "$rank" \
        --world     "$NUM_WORKERS" \
        --batch     "$BATCH" \
        --cache-dir "$CACHE_DIR" \
        --out-dir   "$OUT_DIR" \
        >> "$log" 2>&1 &
    pids+=($!)
    echo "  rank $rank → GPU $gpu, PID $! (log: $log)"
done

echo "모든 워커 시작 완료. 대기 중..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== 완료: $DATASET/$SPLIT ==="
