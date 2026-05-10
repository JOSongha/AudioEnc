#!/bin/bash
# Recreate the five Stage-1 v6 projL model dirs by symlinking weights and
# modeling code from the corresponding base dirs.
#
# Each projL dir owns its own config.json (with adapter_hidden_size=1024,
# intermediate_size=4096, num_attention_heads=16, num_key_value_heads=16
# projector dims, vs. the base's 512/2048/8/8). Everything else
# (safetensors, modeling_*.py, tokenizer, etc.) is symlinked from the base
# so weight files are not duplicated and the base's code is reused verbatim.
#
# Run this once per fresh clone before launching v6 training. Supersedes
# setup_v5_projL_models.sh — adds encodec / wavtok bases.
#
# Prerequisite for encodec / wavtok projL: run scripts/setup_v6_models_weights.sh
# first to populate weight shards for the encodec-24k / wavtok-40-unify bases.

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

# DAC-VAE projL (base lives in jos external/models/)
link_from_base "$EXT/Qwen3.5AE-4B-projL"               "$EXT/Qwen3.5AE-4B"

# Whisper-small projL (base lives in jos external/models/)
link_from_base "$EXT/Qwen3.5AE-4B-whisper-small-projL" "$EXT/Qwen3.5AE-4B-whisper-small"

# Whisper-tiny projL (base lives in jos external/models/)
link_from_base "$EXT/Qwen3.5AE-4B-whisper-tiny-projL"  "$EXT/Qwen3.5AE-4B-whisper-tiny"

# EnCodec / WavTok projL (bases internalised: code/config in jos external/models/,
# weight shards on /mnt/tmp/external/models/ via symlink — see scripts/setup_v6_models_weights.sh)
link_from_base "$EXT/Qwen3.5AE-4B-encodec-projL" "$EXT/Qwen3.5AE-4B-encodec-24k"
link_from_base "$EXT/Qwen3.5AE-4B-wavtok-projL"  "$EXT/Qwen3.5AE-4B-wavtok-40-unify"

echo "[setup] done."
