#!/usr/bin/env bash
# §43 run_retokenize.sh — 8-rank 병렬로 packed arrow 를 src_llm → dst_llm 재토크나이징.
#
# 사용법:
#   bash precompute/run_retokenize.sh \
#       --src-dir /.../mixed/packed_half_inlv_16384 \
#       --src-llm Qwen/Qwen3.5-2B --dst-llm Qwen/Qwen3-1.7B
#
# 출력 dir 은 retokenize_packed.py 가 자동으로 산출 (예: ..._qwen3_1.7b/).

set -euo pipefail

SRC_DIR=""
DST_DIR=""                                   # 생략 시 auto-naming
SRC_LLM="Qwen/Qwen3.5-2B"
DST_LLM=""
CUTOFF_LEN=16384
NUM_RANKS=8
PYTHON="/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-dir)    SRC_DIR="$2";    shift 2 ;;
        --dst-dir)    DST_DIR="$2";    shift 2 ;;
        --src-llm)    SRC_LLM="$2";    shift 2 ;;
        --dst-llm)    DST_LLM="$2";    shift 2 ;;
        --cutoff-len) CUTOFF_LEN="$2"; shift 2 ;;
        --num-ranks)  NUM_RANKS="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$SRC_DIR" || -z "$DST_LLM" ]]; then
    echo "Usage: bash run_retokenize.sh --src-dir <packed_dir> --dst-llm <model> [options]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/retokenize_logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  Retokenize packed arrow (§43)"
echo "  Src dir  : $SRC_DIR"
echo "  Dst dir  : ${DST_DIR:-auto}"
echo "  Src LLM  : $SRC_LLM"
echo "  Dst LLM  : $DST_LLM"
echo "  Cutoff   : $CUTOFF_LEN"
echo "  Ranks    : $NUM_RANKS"
echo "  Log dir  : $LOG_DIR"
echo "========================================"
echo ""

EXTRA_ARGS=""
[[ -n "$DST_DIR" ]] && EXTRA_ARGS="$EXTRA_ARGS --dst-dir $DST_DIR"

PIDS=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
    LOG="$LOG_DIR/retok_rank${rank}.log"
    echo "Launching rank $rank → $LOG"
    "$PYTHON" "$SCRIPT_DIR/retokenize_packed.py" \
        --src-dir "$SRC_DIR" \
        --src-llm "$SRC_LLM" \
        --dst-llm "$DST_LLM" \
        --cutoff-len "$CUTOFF_LEN" \
        --rank "$rank" \
        $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_RANKS ranks launched. Waiting..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  rank $i: done"
    else
        echo "  rank $i: FAILED (see $LOG_DIR/retok_rank${i}.log)"
        FAILED=1
    fi
done

if [[ $FAILED -eq 0 ]]; then
    echo ""
    echo "Retokenize complete."
else
    echo ""
    echo "Some ranks failed. Check logs."
    exit 1
fi
