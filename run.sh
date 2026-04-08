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
# 데이터셋 선택 (기본: ls100,ls360,ls500,mls,gs,vp):
#   bash run.sh --encoder fb_dacvae --datasets ls100,ls360,mls,gs
#
# Stage 1만 실행:
#   bash run.sh --encoder fb_dacvae --stage 1
#
# WandB 비활성화:
#   bash run.sh --encoder fb_dacvae --wandb-mode disabled
#
# conda 환경 경로 재정의:
#   AUDIO_ENV_PATH=/path/to/conda/env bash run.sh --encoder fb_dacvae
#
# 인자 목록:
#   --encoder       필수. encodec | dac | fb_dacvae | mimi_acoustic | mimi_semantic
#   --gpus          GPU 수 (기본 8)
#   --llm           LLM 모델명 (기본: Qwen/Qwen3.5-2B)
#   --datasets      쉼표 구분 (기본: ls100,ls360,ls500,mls,gs,vp)
#   --stage         all | 1 | 2 (기본 all)
#   --stage1-epochs Stage 1 epoch 수
#   --stage2-epochs Stage 2 epoch 수
#   --cutoff-len    Packing 시퀀스 최대 길이 (기본 2048)
#   --eval-steps    WER 평가 주기 (기본: config eval_steps=500)
#   --save-steps    체크포인트 저장 주기 (기본: config save_steps=5000)
#   --attn-impl     eager | sdpa | flash_attention_2 (기본 flash_attention_2)
#   --no-liger      Liger Kernel 비활성화
#   --no-fsdp       FSDP 비활성화
#   --wandb-mode    online | offline | disabled (기본 online)
#   --resume        체크포인트 경로
#   --word-aug      단어 단위 ASR 서브샘플 생성 (word alignment Arrow 사용)
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Conda 환경 설정
# CONDA_PREFIX가 이미 설정된 경우 그대로 사용.
# 미설정 시 AUDIO_ENV_PATH 환경 변수 또는 기본 경로를 사용.
# -----------------------------------------------------------------------------
if [[ -z "$CONDA_PREFIX" ]]; then
    _CONDA_ENV="${AUDIO_ENV_PATH:-/mnt/ddn/users/jos/miniforge3/envs/audio}"
    export PATH="$_CONDA_ENV/bin:$PATH"
    export CONDA_PREFIX="$_CONDA_ENV"
fi

# flash_attn LD_PRELOAD (GLIBCXX_3.4.29 + GLIBC_2.32 우회)
# 자세한 내용: docs/train_pipeline_errors.md §6.7–6.8
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so"

# -----------------------------------------------------------------------------
# 인자 파싱
# -----------------------------------------------------------------------------
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
echo "  Env      : $CONDA_PREFIX"
echo "  Extra    : ${EXTRA_ARGS[*]}"
echo "========================================"

# 기본값: full dataset, sequence packing + FA2 + Liger + FSDP 모두 활성화.
# EXTRA_ARGS에 동일 인자가 있으면 argparse의 last-wins 규칙으로 덮어써짐.
accelerate launch \
    --num_processes "$GPUS" \
    train_pipeline_override.py \
    --encoder "$ENCODER" \
    --datasets ls100,ls360,ls500,mls,gs,vp \
    --liger \
    --fsdp \
    "${EXTRA_ARGS[@]}"
