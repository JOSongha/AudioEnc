#!/usr/bin/env bash
# Extract WAV audio (16 kHz mono) from MELD .mp4 video files, in parallel.
# Input : /mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/{train,dev,test}_splits/*.mp4
# Output: /mnt/tmp/datasets/emotion_raw/MELD/audio/{train,dev,test}/*.wav
set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=/mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw
OUT=/mnt/tmp/datasets/emotion_raw/MELD/audio
LOG=/mnt/tmp/datasets/emotion_raw/MELD/audio_extract.log
mkdir -p "$OUT" && : > "$LOG"
JOBS=${JOBS:-16}
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

extract_one() {
  local src=$1
  local split=$2
  local stem
  stem=$(basename "$src" .mp4)
  local dst=$OUT/$split/${stem}.wav
  [[ -f $dst ]] && return 0
  ffmpeg -nostdin -loglevel error -y -i "$src" -vn -acodec pcm_s16le -ac 1 -ar 16000 "$dst" 2>>"$LOG"
}
export -f extract_one
export OUT LOG

# MELD tarballs extract to non-uniform dir names: train_splits / dev_splits_complete / output_repeated_splits_test
declare -A SPLIT_DIR=( [train]=train_splits [dev]=dev_splits_complete [test]=output_repeated_splits_test )
for split in train dev test; do
  src_dir=$BASE/${SPLIT_DIR[$split]}
  dst_dir=$OUT/$split
  mkdir -p "$dst_dir"
  [[ -d $src_dir ]] || { log "missing $src_dir — skip"; continue; }
  n=$(find "$src_dir" -maxdepth 1 -name "*.mp4" | wc -l)
  log "=== $split: $n mp4 files, JOBS=$JOBS ==="
  find "$src_dir" -maxdepth 1 -name "*.mp4" -print0 \
    | xargs -0 -n1 -P"$JOBS" -I{} bash -c 'extract_one "$@" '"$split" _ {}
  have=$(find "$dst_dir" -maxdepth 1 -name "*.wav" | wc -l)
  log "  $split done: $have wavs"
done

log "=== sizes ==="
du -sh "$OUT"/*/ | tee -a "$LOG"
