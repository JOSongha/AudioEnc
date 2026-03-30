#!/bin/bash
# =============================================================================
# 사용법
# =============================================================================
#
# 기본 (encoder만 지정, 나머지 기본값):
#   bash run.sh --encoder fb_dacvae
#
# GPU 수 변경:
#   bash run.sh --encoder dac --gpus 4
#
# 데이터셋 선택:
#   bash run.sh --encoder fb_dacvae --datasets ls100,ls360,mls,gs
#
# 데이터셋 양 제한:
#   bash run.sh --encoder fb_dacvae \
#       --datasets ls100,ls360,mls \
#       --ls-samples 58000 \
#       --mls-samples 200000
#
# Stage 1 / Stage 2 데이터셋 분리:
#   bash run.sh --encoder fb_dacvae \
#       --s1-datasets ls100,ls360 \
#       --s1-ls-samples 58000 \
#       --datasets ls100,ls360,ls500,mls,gs \
#       --mls-samples 500000
#
# GigaSpeech subset 지정 (xs/s/m/l/xl, 기본 l=2500h):
#   bash run.sh --encoder fb_dacvae --datasets ls100,gs --gs-subset m --gs-samples 100000
#
# 디버그 (step 50에 full weight 저장):
#   bash run.sh --encoder fb_dacvae --debug w
#
# 인자 목록:
#   --encoder       필수. encodec | dac | fb_dacvae | mimi_acoustic | mimi_semantic
#   --gpus          GPU 수 (기본 8)
#   --llm           2b | 4b (기본 2b)
#   --datasets      쉼표 구분. ls100 | ls360 | ls500 | mls | gs (기본: ls100,ls360,ls500,mls)
#   --ls-samples    LibriSpeech 서브샘플 수 (Stage 2)
#   --mls-samples   MLS 샘플 수 (Stage 2)
#   --gs-subset     GigaSpeech 크기: xs(10h) s(250h) m(1000h) l(2500h) xl(10000h)
#   --gs-samples    GigaSpeech 샘플 수 (Stage 2)
#   --s1-datasets   Stage 1 전용 데이터셋 (미지정 시 --datasets 사용)
#   --s1-ls-samples Stage 1 LibriSpeech 서브샘플 수
#   --s1-mls-samples Stage 1 MLS 샘플 수
#   --s1-gs-samples Stage 1 GigaSpeech 샘플 수
#   --wandb-mode    online | offline | disabled
#   --debug         w: step 50에 full weight 저장
# =============================================================================

set -e

ENCODER=""
GPUS=8
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder)  ENCODER="$2"; shift 2 ;;
        --gpus)     GPUS="$2";    shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$ENCODER" ]]; then
    echo "오류: --encoder 를 지정하세요."
    echo "  예) bash run.sh --encoder fb_dacvae"
    exit 1
fi

echo "========================================"
echo "  Encoder  : $ENCODER"
echo "  GPUs     : $GPUS"
echo "  Extra    : ${EXTRA_ARGS[*]}"
echo "========================================"

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    train.py \
    --encoder "$ENCODER" \
    "${EXTRA_ARGS[@]}"
