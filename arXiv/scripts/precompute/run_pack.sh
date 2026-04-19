#!/usr/bin/env bash
# run_pack.sh — 8 rank 병렬 offline packing (GPU 불필요)
#
# 사용법:
#   bash precompute/run_pack.sh --encoder fb_dacvae
#   bash precompute/run_pack.sh --encoder fb_dacvae --datasets ls100,ls360 --cutoff-len 16384

set -euo pipefail

ENCODER=""
DATASETS="ls100,ls360,ls500,mls,gs,vp"
CUTOFF_LEN=""
PRECOMPUTED_DIR="/mnt/ddn/users/jos/precomputed"
NUM_RANKS=8
MIXED=0
WORD_AUG=0
PYTHON="/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)        ENCODER="$2";         shift 2 ;;
        --datasets)       DATASETS="$2";        shift 2 ;;
        --cutoff-len)     CUTOFF_LEN="$2";      shift 2 ;;
        --precomputed-dir) PRECOMPUTED_DIR="$2"; shift 2 ;;
        --num-ranks)      NUM_RANKS="$2";       shift 2 ;;
        --mixed)          MIXED=1;              shift ;;
        --word-aug)       WORD_AUG=1;           shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ENCODER" ]]; then
    echo "Usage: bash run_pack.sh --encoder <encoder_name> [options]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp/pack_logs"
mkdir -p "$LOG_DIR"

EXTRA_ARGS=""
[[ -n "$CUTOFF_LEN" ]] && EXTRA_ARGS="$EXTRA_ARGS --cutoff-len $CUTOFF_LEN"
[[ "$MIXED" -eq 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --mixed"
[[ "$WORD_AUG" -eq 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --word-aug"

echo "========================================"
echo "  Offline packing"
echo "  Encoder  : $ENCODER"
echo "  Ranks    : $NUM_RANKS"
echo "  Datasets : $DATASETS"
echo "  Cutoff   : ${CUTOFF_LEN:-from config}"
echo "  Mixed    : $([ "$MIXED" -eq 1 ] && echo '✓' || echo '✗')"
echo "  Word-aug : $([ "$WORD_AUG" -eq 1 ] && echo '✓' || echo '✗')"
echo "========================================"
echo ""

PIDS=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
    LOG="$LOG_DIR/pack_rank${rank}.log"
    echo "Launching rank $rank → $LOG"
    "$PYTHON" "$SCRIPT_DIR/pack_arrow.py" \
        --encoder "$ENCODER" \
        --rank "$rank" \
        --num-ranks "$NUM_RANKS" \
        --datasets "$DATASETS" \
        --precomputed-dir "$PRECOMPUTED_DIR" \
        $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_RANKS ranks launched. Waiting..."
echo "  Logs: $LOG_DIR/pack_rank{0..7}.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  rank $i: done"
    else
        echo "  rank $i: FAILED (see $LOG_DIR/pack_rank${i}.log)"
        FAILED=1
    fi
done

if [[ $FAILED -eq 0 ]]; then
    echo ""
    echo "Packing complete."
    # mixed 모드는 자동 rebalance 로 rank 간 bin 개수 불일치 제거.
    # (per-dataset packed 모드는 해당 없음 — 학습 시 rank 간 독립 로드)
    if [[ "$MIXED" -eq 1 ]]; then
        echo ""
        echo "Running rebalance (mixed rank-balance)..."
        "$PYTHON" "$SCRIPT_DIR/pack_arrow.py" \
            --encoder "$ENCODER" \
            --num-ranks "$NUM_RANKS" \
            --precomputed-dir "$PRECOMPUTED_DIR" \
            ${CUTOFF_LEN:+--cutoff-len $CUTOFF_LEN} \
            --rebalance
    fi
else
    echo ""
    echo "Some ranks failed. Check logs."
    exit 1
fi
