#!/bin/bash
# Periodically hardlink newly-saved checkpoint-* dirs from SRC into DST so
# they survive HF Trainer's save_total_limit rotation.
#
# Same-filesystem hardlink — zero extra disk. Idempotent. Waits for save
# to finish (mtime > $MIN_AGE_SEC) before linking.

set -u

SRC=${SRC:-/mnt/tmp/results/Qwen3.5AE-ASR-Stage1-libri_mls_vox}
DST=${DST:-/mnt/tmp/results/Qwen3.5AE-ASR-Stage1-libri_mls_vox_archive}
MIN_AGE_SEC=${MIN_AGE_SEC:-60}
POLL_INTERVAL=${POLL_INTERVAL:-30}

mkdir -p "$DST"
echo "[watcher] SRC=$SRC"
echo "[watcher] DST=$DST"
echo "[watcher] min_age=${MIN_AGE_SEC}s  poll=${POLL_INTERVAL}s"

while true; do
    for d in "$SRC"/checkpoint-*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        dst="$DST/$name"
        [ -e "$dst" ] && continue
        # Wait until safetensors save is finished — use the index file's mtime
        ref="$d/model.safetensors.index.json"
        [ -f "$ref" ] || ref="$d/model.safetensors"
        [ -f "$ref" ] || continue
        mtime=$(stat -c %Y "$ref" 2>/dev/null) || continue
        now=$(date +%s)
        age=$((now - mtime))
        if [ "$age" -lt "$MIN_AGE_SEC" ]; then
            echo "[watcher] $(date +%T) $name: save in progress (age=${age}s), waiting"
            continue
        fi
        cp -al "$d" "${dst}.tmp" && mv "${dst}.tmp" "$dst" \
            && echo "[watcher] $(date +%T) archived $name" \
            || echo "[watcher] $(date +%T) FAILED $name"
    done
    sleep "$POLL_INTERVAL"
done
