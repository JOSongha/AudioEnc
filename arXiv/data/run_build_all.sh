#!/bin/bash
# 모든 데이터셋 Arrow 변환 실행
# Usage: bash run_build_all.sh [--skip-ls] [--skip-gs] [--skip-vp]
#
# 각 작업은 독립적으로 백그라운드 실행, 로그는 /mnt/tmp/cache/arrow_build_*.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/mnt/fr20tb/wbl_residency/jos/.venv310/bin/python3"
CACHE_DIR="/mnt/tmp/cache"
LS_ROOT="$CACHE_DIR/LibriSpeech"
LOG_DIR="$CACHE_DIR"

SKIP_LS=false
SKIP_GS=false
SKIP_VP=false

for arg in "$@"; do
    case $arg in
        --skip-ls) SKIP_LS=true ;;
        --skip-gs) SKIP_GS=true ;;
        --skip-vp) SKIP_VP=true ;;
    esac
done

PIDS=()

# ── LibriSpeech ──────────────────────────────────────────────────────────────
if ! $SKIP_LS; then
    echo "[LibriSpeech] Arrow 변환 시작..."
    nohup $VENV "$SCRIPT_DIR/build_librispeech_arrow.py" \
        --librispeech-root "$LS_ROOT" \
        --output-dir       "$CACHE_DIR" \
        --num-proc         16 \
        >> "$LOG_DIR/arrow_build_librispeech.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}  로그: $LOG_DIR/arrow_build_librispeech.log"
fi

# ── GigaSpeech XL ────────────────────────────────────────────────────────────
if ! $SKIP_GS; then
    echo "[GigaSpeech XL] Arrow 빌딩 시작..."
    nohup $VENV "$SCRIPT_DIR/build_gigaspeech_arrow.py" \
        --cache-dir "$CACHE_DIR" \
        --subset    xl \
        --num-proc  16 \
        >> "$LOG_DIR/arrow_build_gigaspeech.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}  로그: $LOG_DIR/arrow_build_gigaspeech.log"
fi

# ── VoxPopuli EN ─────────────────────────────────────────────────────────────
if ! $SKIP_VP; then
    echo "[VoxPopuli EN] 다운로드 및 Arrow 빌딩 시작..."
    nohup $VENV "$SCRIPT_DIR/build_voxpopuli_arrow.py" \
        --cache-dir "$CACHE_DIR" \
        --num-proc  8 \
        >> "$LOG_DIR/arrow_build_voxpopuli.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}  로그: $LOG_DIR/arrow_build_voxpopuli.log"
fi

echo ""
echo "실행 중인 작업: ${#PIDS[@]}개"
echo "진행 확인:"
echo "  tail -f $LOG_DIR/arrow_build_librispeech.log"
echo "  tail -f $LOG_DIR/arrow_build_gigaspeech.log"
echo "  tail -f $LOG_DIR/arrow_build_voxpopuli.log"
