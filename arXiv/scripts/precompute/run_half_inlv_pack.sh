#!/usr/bin/env bash
# run_half_inlv_pack.sh — half-interleave offline packing (8 rank 병렬, GPU 불필요)
#
# 사용법:
#   bash precompute/run_half_inlv_pack.sh --encoder fb_dacvae
#   bash precompute/run_half_inlv_pack.sh --encoder fb_dacvae --datasets ls100,ls360 --cutoff-len 16384

set -euo pipefail

ENCODER=""
DATASETS="ls100,ls360,ls500,mls,gs,vp"
CUTOFF_LEN=""
PRECOMPUTED_DIR="/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/precomputed"
NUM_RANKS=8
NUM_WORKERS=1                                # rank 내부 phase 1 병렬도 (기본 sequential)
SHARDS_PER_RANK=8                            # rank 당 출력 shard 개수 (equal bin)
MIXED=1                                      # 기본 mixed (cross-dataset shuffle)
SENTENCE_ONLY=0                              # §42: interleave 끄고 sentence-only
LLM=""                                       # §43 --llm 오버라이드 (비우면 config 기본값)
PYTHON="/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)         ENCODER="$2";         shift 2 ;;
        --datasets)        DATASETS="$2";        shift 2 ;;
        --cutoff-len)      CUTOFF_LEN="$2";      shift 2 ;;
        --precomputed-dir) PRECOMPUTED_DIR="$2"; shift 2 ;;
        --num-ranks)       NUM_RANKS="$2";       shift 2 ;;
        --num-workers)     NUM_WORKERS="$2";     shift 2 ;;
        --shards-per-rank) SHARDS_PER_RANK="$2"; shift 2 ;;
        --no-mixed)        MIXED=0;              shift ;;
        --sentence-only)   SENTENCE_ONLY=1;      shift ;;
        --llm)             LLM="$2";             shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ENCODER" ]]; then
    echo "Usage: bash run_half_inlv_pack.sh --encoder <encoder_name> [options]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/pack_logs_half_inlv"
mkdir -p "$LOG_DIR"

EXTRA_ARGS=""
[[ -n "$CUTOFF_LEN" ]] && EXTRA_ARGS="$EXTRA_ARGS --cutoff-len $CUTOFF_LEN"
[[ "$MIXED" -eq 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --mixed"
[[ "$NUM_WORKERS" -gt 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --num-workers $NUM_WORKERS"
EXTRA_ARGS="$EXTRA_ARGS --shards-per-rank $SHARDS_PER_RANK"
[[ "$SENTENCE_ONLY" -eq 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --sentence-only"
[[ -n "$LLM" ]] && EXTRA_ARGS="$EXTRA_ARGS --llm $LLM"

echo "========================================"
echo "  Half-interleave offline packing"
echo "  Encoder  : $ENCODER"
echo "  Ranks    : $NUM_RANKS"
echo "  Workers  : $NUM_WORKERS (rank 내부 phase 1 병렬도)"
echo "  Shards/r : $SHARDS_PER_RANK (equal bin count)"
echo "  Datasets : $DATASETS"
echo "  Precomp  : $PRECOMPUTED_DIR"
echo "  Cutoff   : ${CUTOFF_LEN:-from config}"
echo "  LLM      : ${LLM:-from config default}"
echo "  Mixed    : $([ "$MIXED" -eq 1 ] && echo '✓' || echo '✗')"
echo "  Log dir  : $LOG_DIR"
echo "========================================"
echo ""

PIDS=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
    LOG="$LOG_DIR/pack_rank${rank}.log"
    echo "Launching rank $rank → $LOG"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "$SCRIPT_DIR/half_inlv_pack_arrow.py" \
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
echo "  Logs: $LOG_DIR/pack_rank{0..$((NUM_RANKS-1))}.log"
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
    if [[ "$MIXED" -eq 1 ]]; then
        echo ""
        echo "Running rebalance (mixed rank-balance)..."
        CUDA_VISIBLE_DEVICES="" "$PYTHON" "$SCRIPT_DIR/half_inlv_pack_arrow.py" \
            --encoder "$ENCODER" \
            --num-ranks "$NUM_RANKS" \
            --precomputed-dir "$PRECOMPUTED_DIR" \
            ${CUTOFF_LEN:+--cutoff-len $CUTOFF_LEN} \
            $([[ "$SENTENCE_ONLY" -eq 1 ]] && echo "--sentence-only") \
            ${LLM:+--llm $LLM} \
            --rebalance
    fi
else
    echo ""
    echo "Some ranks failed. Check logs."
    exit 1
fi
