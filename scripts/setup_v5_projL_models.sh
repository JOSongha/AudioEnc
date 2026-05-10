#!/bin/bash
# Recreate the three Stage-1 v5 projL model dirs by symlinking weights and
# modeling code from the corresponding base dirs.
#
# Each projL dir owns its own config.json (with H=1024/I=4096/heads=16
# projector dims, vs. the base's H=512/I=2048/heads=8). Everything else
# (safetensors, modeling_*.py, tokenizer, etc.) is symlinked from the base
# so weight files are not duplicated and the base's code is reused verbatim.
#
# Run this once per fresh clone before launching v5 training.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$HERE/../external/models"

link_from_base() {
    local projl="$1"
    local base="$2"
    if [ ! -d "$base" ]; then
        echo "[setup] skip $projl (base $base missing)"
        return
    fi
    if [ ! -f "$projl/config.json" ]; then
        echo "[setup] ERROR: $projl/config.json missing — projL config not yet committed?"
        return 1
    fi
    for f in "$base"/*; do
        local name; name=$(basename "$f")
        if [ "$name" = "config.json" ]; then continue; fi
        ln -sf "$f" "$projl/$name"
    done
    echo "[setup] $projl  ← linked from $base"
}

link_from_base "$EXT/Qwen3.5AE-4B-projL"               "$EXT/Qwen3.5AE-4B"
link_from_base "$EXT/Qwen3.5AE-4B-whisper-small-projL" "$EXT/Qwen3.5AE-4B-whisper-small"

# Whisper-tiny base may live in jos or sehyun's external/models — try both.
TINY_PROJL="$EXT/Qwen3.5AE-4B-whisper-tiny-projL"
for cand in \
    "$EXT/Qwen3.5AE-4B-whisper-tiny" \
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/models/Qwen3.5AE-4B-whisper-tiny"
do
    if [ -d "$cand" ]; then
        link_from_base "$TINY_PROJL" "$cand"
        break
    fi
done

echo "[setup] done."
