#!/bin/bash
# Bootstrap external/models/ from nubes archive.
#
# Repo clone gets the code + config files but not the safetensors weights
# (.gitignore excludes external/models/**/*.safetensors). This script fetches
# the 10 base safetensors (5 encoder variants × 2 shards) from the nubes
# archive at users/jos/AudioEnc/models/, then regenerates the projL overlay
# symlinks with repo-relative paths so HF transformers' from_pretrained sees
# each projL dir as a complete model.
#
# Usage (run from repo root):
#   bash scripts/setup_models.sh
#
# Resume-safe: skips files whose local size already matches the nubes
# `X-Object-Size`. Re-run any time.

set -u
set -o pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

export NUBES_GATEWAY_ADDRESS="${NUBES_GATEWAY_ADDRESS:-c.nubes.sto.navercorp.com:8000}"
export NUBES_IP_LOOKUP_ADDRESS="${NUBES_IP_LOOKUP_ADDRESS:-c.lookup.nubes.navercorp.com:8080}"
NC="${NUBESCLI:-/mnt/ddn/users/jos/cli/nubescli}"
NUBES_ROOT="hyperscaleai-audiollm/users/jos/AudioEnc/models"

if [ ! -x "$NC" ]; then
    echo "[setup_models] nubescli not found at $NC" >&2
    echo "[setup_models] override with NUBESCLI=/path/to/nubescli bash scripts/setup_models.sh" >&2
    exit 1
fi

# Encoder base dirs to fetch + the projL overlay that uses each.
# Format: "<base_dir> <projL_overlay_dir>"
BASE_OVERLAY_PAIRS=(
    "Qwen3.5AE-4B                  Qwen3.5AE-4B-projL"
    "Qwen3.5AE-4B-encodec-24k      Qwen3.5AE-4B-encodec-projL"
    "Qwen3.5AE-4B-wavtok-40-unify  Qwen3.5AE-4B-wavtok-projL"
    "Qwen3.5AE-4B-whisper-tiny     Qwen3.5AE-4B-whisper-tiny-projL"
    "Qwen3.5AE-4B-whisper-small    Qwen3.5AE-4B-whisper-small-projL"
)
SHARDS=("model-00001-of-00002.safetensors" "model-00002-of-00002.safetensors")

echo "[setup_models] repo root: $REPO"
echo "[setup_models] nubes:     $NUBES_ROOT"
echo

# ── 1. Fetch safetensors from nubes (10 files, ~40 GB total) ──────────────────
echo "[setup_models] step 1/2: fetching safetensors from nubes"
for pair in "${BASE_OVERLAY_PAIRS[@]}"; do
    read base _overlay <<< "$pair"
    dest_dir="external/models/$base"
    mkdir -p "$dest_dir"
    for shard in "${SHARDS[@]}"; do
        local_path="$dest_dir/$shard"
        nubes_path="$NUBES_ROOT/$base/$shard"

        # Skip if file already exists with non-zero size
        if [ -f "$local_path" ] && [ ! -L "$local_path" ] && [ -s "$local_path" ]; then
            local_size=$(stat -c %s "$local_path")
            nubes_size=$("$NC" status "$nubes_path" 2>/dev/null | awk -F: '/X-Object-Size/ {gsub(" ",""); print $2}')
            if [ "$local_size" = "$nubes_size" ]; then
                printf '  [skip ] %-50s (%s)\n' "$base/$shard" "$(numfmt --to=iec --suffix=B "$local_size")"
                continue
            fi
        fi

        # Remove any stale symlink before downloading
        [ -L "$local_path" ] && rm -f "$local_path"

        printf '  [fetch] %-50s ' "$base/$shard"
        "$NC" download -w -n "$nubes_path" "$local_path" >/dev/null 2>&1
        if [ -s "$local_path" ]; then
            echo "ok ($(du -h "$local_path" | cut -f1))"
        else
            echo "FAILED"
            exit 1
        fi
    done
done
echo

# ── 2. Regenerate projL overlay symlinks (relative paths) ─────────────────────
echo "[setup_models] step 2/2: regenerating projL overlay symlinks"
for pair in "${BASE_OVERLAY_PAIRS[@]}"; do
    read base overlay <<< "$pair"
    base_dir="external/models/$base"
    overlay_dir="external/models/$overlay"
    if [ ! -d "$overlay_dir" ]; then
        echo "  [skip ] $overlay (overlay dir missing — config.json should be in git)"
        continue
    fi

    # For every file in base, create a relative symlink in overlay, except
    # config.json which is overlay-specific (projL audio_config) and stays as-is.
    n_link=0
    for src in "$base_dir"/*; do
        fname=$(basename "$src")
        [ "$fname" = "config.json" ] && continue
        [ "$fname" = "__pycache__" ] && continue
        dst="$overlay_dir/$fname"
        # Remove any stale entry then create relative symlink (../<base>/<fname>)
        rm -f "$dst" 2>/dev/null || true
        ln -s "../$base/$fname" "$dst"
        n_link=$((n_link + 1))
    done
    printf '  [link ] %-40s %d files -> ../%s/\n' "$overlay" "$n_link" "$base"
done

echo
echo "[setup_models] done."
echo "[setup_models] verify: ls -la external/models/Qwen3.5AE-4B-projL/ | head"
