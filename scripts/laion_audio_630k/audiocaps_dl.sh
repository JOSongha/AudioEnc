#!/bin/bash
# AudioCaps via OpenSound/AudioCaps HF mirror.
# Note: AudioCaps is NOT part of LAION-Audio-630K — included here for convenience.
# 51,812 clips (train 45,178 / val 2,223 / test 4,411), audio embedded in parquet.
# ~44 GB, 41 train + 2 val + 4 test parquets.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-/mnt/tmp/cache/huggingface}
OUT=${OUT:-/mnt/tmp/datasets/audiocaps}
huggingface-cli download OpenSound/AudioCaps \
    --repo-type dataset \
    --local-dir "$OUT" \
    --max-workers 16
