#!/bin/bash
# =============================================================================
# 사용법
# =============================================================================
#
# 기본:
#   bash run_debug.sh --encoder fb_dacvae
#
# CTC loss 추가:
#   bash run_debug.sh --encoder fb_dacvae --debug c
#
# GPU / port 지정:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_debug.sh --encoder fb_dacvae --debug c --port 29502
#
# LLM 변경:
#   bash run_debug.sh --encoder fb_dacvae --llm Qwen/Qwen3.5-4B-Instruct
#
# 에폭 수 조정:
#   bash run_debug.sh --encoder fb_dacvae --stage1-epochs 3 --stage2-epochs 3
#
# 저장 경로 변경:
#   bash run_debug.sh --encoder fb_dacvae --debug-dir /mnt/tmp/my_run
#
# WER 샘플 수 / 출력 수 조정:
#   bash run_debug.sh --encoder fb_dacvae --wer-samples 100 --print-samples 5
#
# wandb 비활성화:
#   bash run_debug.sh --encoder fb_dacvae --wandb-mode disabled
#
# 인자 목록:
#   --encoder           필수. encodec | dac | fb_dacvae | mimi_acoustic | mimi_semantic
#   --gpus              GPU 수 (기본: CUDA_VISIBLE_DEVICES 개수, 없으면 1)
#   --port              torchrun master_port (기본 29501)
#   --debug             c: CTC loss 추가, e: EOS loss upweighting, d: EOS weight linear decay (e와 함께 사용)
#   --llm               LLM 모델 이름 (기본: config 값, 예: Qwen/Qwen3.5-0.8B)
#   --stage1-epochs     Stage 1 에폭 수 (기본 5)
#   --stage2-epochs     Stage 2 에폭 수 (기본 5)
#   --debug-dir         체크포인트·CSV 저장 경로 (기본: model_cache_dir/debug_<encoder>_<ctc|nonctc>)
#   --wandb-mode        online | offline | disabled (기본 online)
#   --wer-samples       에폭당 WER 평가 샘플 수 (기본 200)
#   --print-samples     에폭당 REF/HYP 출력 샘플 수 (기본 10)
#   --div               데이터셋 서브셋 분모 (기본 5 → 1/5 사용)
#   --eos-weight        EOS 토큰 loss 가중치 (debug=e일 때 적용, 기본 3.0)
#   --eos-decay-epochs  EOS weight를 1.0까지 decay할 에폭 수 (debug=ed, 기본: 전체 에폭)
# =============================================================================

set -e

ENCODER="fb_dacvae"
PORT=29501
EXTRA_ARGS=()

# CUDA_VISIBLE_DEVICES에서 GPU 수 자동 감지
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    GPUS=1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder)  ENCODER="$2"; shift 2 ;;
        --gpus)     GPUS="$2";    shift 2 ;;
        --port)     PORT="$2";    shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$ENCODER" ]]; then
    echo "오류: --encoder 를 지정하세요."
    echo "  예) bash run_debug.sh --encoder fb_dacvae --debug ce --port 29502"
    exit 1
fi

echo "========================================"
echo "  Encoder  : $ENCODER"
echo "  GPUs     : $GPUS"
echo "  Port     : $PORT"
echo "  Extra    : ${EXTRA_ARGS[*]}"
echo "========================================"

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train_debug.py \
    --encoder "$ENCODER" \
    "${EXTRA_ARGS[@]}"


# CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_debug.sh --debug ced --port 29502 --div 5 --eos-decay-epochs 20