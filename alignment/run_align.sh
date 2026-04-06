#!/bin/bash
# LibriSpeech word alignment — 8 GPU 병렬 실행
# Usage: bash run_align.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/mnt/fr20tb/wbl_residency/jos/.venv310/bin/python3"
LS_ROOT="/mnt/tmp/cache/LibriSpeech"
OUT_DIR="/mnt/tmp/cache/word_alignments"
NUM_GPUS=8

echo "=== Step 1: manifest 생성 ==="
$VENV "$SCRIPT_DIR/build_manifest.py" \
    --librispeech-root "$LS_ROOT" \
    --output-dir "$OUT_DIR" \
    --num-shards "$NUM_GPUS"

echo ""
echo "=== Step 2: 8 GPU 병렬 alignment ==="

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i $VENV "$SCRIPT_DIR/align_worker.py" \
        --shard "$OUT_DIR/shard_${i}.jsonl" \
        --output-dir "$OUT_DIR" \
        --gpu 0 \
        --rank $i \
        > "$OUT_DIR/worker_${i}.stdout" 2>&1 &
    PIDS+=($!)
    echo "  GPU $i 시작 (PID ${PIDS[-1]})"
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
        echo "  GPU $i 완료 (PID $pid)"
    else
        echo "  GPU $i 실패 (PID $pid, exit code $?)"
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "=== 전체 완료 ==="
else
    echo "=== 일부 실패 — $OUT_DIR/worker_N.log 확인 ==="
fi
