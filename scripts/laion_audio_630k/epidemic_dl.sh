#!/bin/bash
# Epidemic Sound via CLAPv2/epidemic_sound_effects_t5_debiased HF mirror.
# Audio embedded in parquet (FLAC bytes), bypassing dead cloudfront URLs.
# 72 GB / 2524 parquets / ~75k clips / T5 debiased captions in `text` column.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-/mnt/tmp/cache/huggingface}
OUT=${OUT:-/mnt/tmp/datasets/laion_epidemic_clapv2}
huggingface-cli download CLAPv2/epidemic_sound_effects_t5_debiased \
    --repo-type dataset \
    --local-dir "$OUT" \
    --max-workers 16
