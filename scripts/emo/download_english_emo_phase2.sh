#!/usr/bin/env bash
# Phase 2 for open English emo: fetch actual data after phase 1 repo inspection.
#   - EmoV-DB via OpenSLR 115 (4 speakers × up-to-5 emotions per-tar)
#   - DailyTalk via gdown on upstream Google Drive folder
set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download_english_emo_phase2.log
: > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

# --- EmoV-DB (OpenSLR 115) ---------------------------------------------------
emov() {
  log "=== EmoV-DB via OpenSLR 115 ==="
  local dir=$BASE/EmoV-DB
  mkdir -p "$dir"
  local urls=(
    bea_Amused bea_Angry bea_Disgusted bea_Neutral bea_Sleepy
    jenie_Amused jenie_Angry jenie_Disgusted jenie_Neutral jenie_Sleepy
    josh_Amused josh_Neutral josh_Sleepy
    sam_Amused sam_Angry sam_Disgusted sam_Neutral sam_Sleepy
  )
  for stem in "${urls[@]}"; do
    local out=$dir/${stem}.tar.gz
    local url="https://www.openslr.org/resources/115/${stem}.tar.gz"
    if [[ -s $out ]]; then
      log "  $stem already — skip"
    else
      wget --no-verbose -c --timeout=120 -O "$out" "$url" 2>&1 | tee -a "$LOG"
    fi
  done
  log "  extracting tarballs…"
  for t in "$dir"/*.tar.gz; do
    (cd "$dir" && tar -xzf "$(basename "$t")") 2>&1 | tail -2 | tee -a "$LOG"
  done
  log "  EmoV-DB final:" ; du -sh "$dir" | tee -a "$LOG"
  local n=$(find "$dir" -name "*.wav" 2>/dev/null | wc -l)
  log "  EmoV-DB wav count: $n"
}

# --- DailyTalk (Google Drive folder) -----------------------------------------
dailytalk() {
  log "=== DailyTalk via gdown (GDrive folder) ==="
  local dir=$BASE/DailyTalk
  mkdir -p "$dir"
  # skip if dataset dir already populated
  if [[ -n $(find "$dir" -maxdepth 2 -name "*.wav" 2>/dev/null | head -1) ]]; then
    log "  already populated — skip"; return
  fi
  local folder_id='1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL'
  (cd "$dir" && gdown --folder "https://drive.google.com/drive/folders/${folder_id}" -O .) 2>&1 | tail -30 | tee -a "$LOG"
  # gdown drops dailytalk.zip at top level; unzip it if present.
  if [[ -s $dir/dailytalk.zip ]]; then
    log "  unzipping dailytalk.zip"
    (cd "$dir" && unzip -q -n dailytalk.zip) 2>&1 | tail -3 | tee -a "$LOG"
  fi
  log "  DailyTalk final:" ; du -sh "$dir" | tee -a "$LOG"
  local n=$(find "$dir" -name "*.wav" 2>/dev/null | wc -l)
  log "  DailyTalk wav count: $n"
}

main() {
  emov
  dailytalk
  log "=== sizes ==="
  du -sh "$BASE/EmoV-DB" "$BASE/DailyTalk" 2>/dev/null | tee -a "$LOG"
}

main "$@"
