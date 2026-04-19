#!/bin/bash
# =============================================================================
# run_precompute.sh — 8 GPU 병렬 인코더 피처 사전 계산
#
# 사용법:
#   bash precompute/run_precompute.sh --encoder fb_dacvae
#   bash precompute/run_precompute.sh --encoder fb_dacvae --datasets ls100,gs
#   bash precompute/run_precompute.sh --encoder fb_dacvae --gpus 4
#   bash precompute/run_precompute.sh --encoder fb_dacvae --verify
#   bash precompute/run_precompute.sh --encoder fb_dacvae --batch-size 50
#
# 각 GPU가 독립 프로세스로 실행되며 데이터셋을 1/N 샤드씩 처리한다.
# 결과: /mnt/fr20tb/wbl_residency/jos/ddn/precomputed/{encoder}/{dataset}/rank{k}.arrow
# =============================================================================

set -e

# Ctrl-C / 종료 시 모든 child 프로세스 강제 종료
_cleanup() {
    echo "[run_precompute] 종료 신호 수신, child 프로세스 정리 중..."
    pkill -9 -P $$ 2>/dev/null || true
    pkill -9 -f precompute_features.py 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# Conda 환경 설정
if [[ -z "$CONDA_PREFIX" ]]; then
    _CONDA_ENV="${AUDIO_ENV_PATH:-/mnt/ddn/users/jos/miniforge3/envs/audio}"
    export PATH="$_CONDA_ENV/bin:$PATH"
    export CONDA_PREFIX="$_CONDA_ENV"
fi

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so"

# -----------------------------------------------------------------------------
# 인자 파싱
# -----------------------------------------------------------------------------
ENCODER=""
GPUS=8
GPU_IDS=""   # 쉼표 구분 GPU ID 목록 (예: "0,2,3,5,6,7"). 미지정 시 0..GPUS-1 사용.
VERIFY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder)  ENCODER="$2"; shift 2 ;;
        --gpus)     GPUS="$2";    shift 2 ;;
        --gpu-ids)  GPU_IDS="$2"; shift 2 ;;
        --verify)   VERIFY=1;     shift   ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# GPU ID 배열 구성
if [[ -n "$GPU_IDS" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
    GPUS=${#GPU_LIST[@]}
else
    GPU_LIST=()
    for (( i=0; i<GPUS; i++ )); do GPU_LIST+=($i); done
fi

if [[ -z "$ENCODER" ]]; then
    echo "오류: --encoder 를 지정하세요."
    echo "  예) bash precompute/run_precompute.sh --encoder fb_dacvae"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRECOMPUTE_SCRIPT="$SCRIPT_DIR/precompute_features.py"

# -----------------------------------------------------------------------------
# 완료 확인 모드
# -----------------------------------------------------------------------------
if [[ $VERIFY -eq 1 ]]; then
    python "$PRECOMPUTE_SCRIPT" --encoder "$ENCODER" --verify
    exit 0
fi

echo "========================================"
echo "  Encoder  : $ENCODER"
echo "  GPUs     : $GPUS"
echo "  Extra    : ${EXTRA_ARGS[*]}"
echo "========================================"

# -----------------------------------------------------------------------------
# GPU별 병렬 실행
# -----------------------------------------------------------------------------
PIDS=()
LOG_DIR="/tmp/precompute_logs/${ENCODER}"
mkdir -p "$LOG_DIR"

for (( IDX=0; IDX<GPUS; IDX++ )); do
    GPU_ID=${GPU_LIST[$IDX]}
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"
    echo "  GPU $GPU_ID (rank $IDX) 시작 → $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
        python "$PRECOMPUTE_SCRIPT" \
            --encoder  "$ENCODER" \
            --gpu-id   "$IDX" \
            --num-gpus "$GPUS" \
            "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "로그 실시간 확인:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu*.log"
echo ""

# 모든 프로세스 완료 대기
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "  GPU $i 완료 (PID $PID)"
    else
        echo "  GPU $i 실패 (PID $PID, exit $?)"
        FAILED=$((FAILED+1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "모든 GPU 전처리 완료."
    echo "확인: bash precompute/run_precompute.sh --encoder $ENCODER --verify"
else
    echo "경고: $FAILED 개 GPU 실패. 로그 확인:"
    echo "  ls $LOG_DIR/"
    exit 1
fi
