#!/bin/bash
# 사용법:
#   bash run.sh encodec
#   bash run.sh dac
#   bash run.sh dac_vae
#   bash run.sh mimi_acoustic
#   bash run.sh mimi_semantic

set -e

ENCODER=${1:?"encoder를 지정하세요: bash run.sh [encodec|dac|mimi_acoustic|mimi_semantic]"}
N_GPU=${2:-8}   # 두 번째 인자로 GPU 수 변경 가능 (기본 8)

echo "========================================"
echo "  Encoder : $ENCODER"
echo "  GPUs    : $N_GPU"
echo "========================================"

torchrun \
    --nproc_per_node=$N_GPU \
    --master_port=29500 \
    train.py \
    --encoder "$ENCODER"
