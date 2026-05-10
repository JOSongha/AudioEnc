#!/bin/bash
# Internalise EnCodec / WavTok base model weights (~16 GB total).
#
# Layout:
#   - Code/config (audio_encoder.py, modeling_qwen3_5AE.py, tokenizer.json, ...) lives
#     in jos external/models/Qwen3.5AE-4B-{encodec-24k,wavtok-40-unify}/ (git-tracked).
#   - Weight shards (*.safetensors, ~8 GB each) live on /mnt/tmp/external/models/
#     and are symlinked into the jos base dir (gitignored).
#
# Run this once per fresh checkout, before scripts/setup_v6_projL_models.sh.
#
# Source paths default to sehyun's audiollm-trainer external/models. Override
# SEHYUN_EXT to point elsewhere (e.g. an HF cache) if needed.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$HERE/../external/models"
TMP_EXT=/mnt/tmp/external/models
SEHYUN_EXT="${SEHYUN_EXT:-/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/models}"

setup_weights() {
    local name="$1"  # e.g. Qwen3.5AE-4B-encodec-24k
    local src="$SEHYUN_EXT/$name"
    local tmp="$TMP_EXT/$name"
    local dst="$EXT/$name"

    if [ ! -d "$src" ]; then
        echo "[weights] skip $name (source $src missing)"
        return
    fi
    mkdir -p "$tmp" "$dst"

    # 1. rsync safetensors to /mnt/tmp (~8 GB per encoder)
    rsync -aH --info=progress2 "$src"/*.safetensors "$tmp/"

    # 2. symlink jos base dir's safetensors → /mnt/tmp copy
    for f in "$tmp"/*.safetensors; do
        local base; base=$(basename "$f")
        ln -sf "$f" "$dst/$base"
    done

    echo "[weights] $name  ← weights on $tmp, symlinked into $dst"
}

setup_weights Qwen3.5AE-4B-encodec-24k
setup_weights Qwen3.5AE-4B-wavtok-40-unify

echo "[weights] done."
