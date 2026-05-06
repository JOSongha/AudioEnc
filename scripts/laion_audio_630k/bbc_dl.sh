#!/bin/bash
# BBC Sound Effects via marianna13/BBCSoundEffects HF mirror.
# Original LAION csv → BBC Rewind SPA (broken). Mirror is 166 GB / 66 tar shards.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-/mnt/tmp/cache/huggingface}
OUT=${OUT:-/mnt/tmp/datasets/laion_bbc_hf}
huggingface-cli download marianna13/BBCSoundEffects \
    --repo-type dataset \
    --local-dir "$OUT" \
    --max-workers 16
