#!/bin/bash
# LibriSpeech word alignment — 다중 GPU × 다중 워커 병렬 실행
# Usage: bash run_align.sh [WORKERS_PER_GPU]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/mnt/fr20tb/wbl_residency/jos/.venv310/bin/python3"
LS_ROOT="/mnt/tmp/cache/LibriSpeech"
OUT_DIR="/mnt/tmp/cache/word_alignments"
NUM_GPUS=8
WORKERS_PER_GPU="${1:-8}"   # 기본 8개/GPU → 총 64 워커
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

echo "=== Step 1: manifest 생성 (${TOTAL_WORKERS} shards) ==="
$VENV "$SCRIPT_DIR/build_manifest.py" \
    --librispeech-root "$LS_ROOT" \
    --output-dir "$OUT_DIR" \
    --num-shards "$TOTAL_WORKERS"

echo ""
echo "=== Step 2: ${NUM_GPUS} GPU × ${WORKERS_PER_GPU} workers = ${TOTAL_WORKERS} 병렬 alignment ==="

PIDS=()
rank=0
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for w in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu $VENV "$SCRIPT_DIR/align_worker.py" \
            --shard "$OUT_DIR/shard_${rank}.jsonl" \
            --output-dir "$OUT_DIR" \
            --gpu 0 \
            --rank $rank \
            > "$OUT_DIR/worker_${rank}.stdout" 2>&1 &
        PIDS+=($!)
        echo "  GPU $gpu worker $w  rank=$rank  PID=${PIDS[-1]}"
        rank=$((rank + 1))
    done
done

echo ""
echo "모든 워커 실행 중. 로그: $OUT_DIR/worker_N.log"
echo "진행 확인: tail -f $OUT_DIR/worker_0.log"
echo ""

# 모든 워커 완료 대기
ALL_OK=true
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait $pid; then
        echo "  rank $i 완료 (PID $pid)"
    else
        echo "  rank $i 실패 (PID $pid, exit code $?)"
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "=== 전체 완료 ==="
    echo ""
    echo "=== Step 3: 병합 및 검증 ==="
    $VENV "$SCRIPT_DIR/merge_alignments.py" \
        --alignment-dir "$OUT_DIR" \
        --manifest "$OUT_DIR/manifest.jsonl" \
        --output-dir "$OUT_DIR"
else
    echo "=== 일부 실패 — $OUT_DIR/worker_N.log 확인 ==="
fi
