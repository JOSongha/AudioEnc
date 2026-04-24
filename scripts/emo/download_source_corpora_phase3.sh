#!/usr/bin/env bash
# Phase 3: LFS pull + GDrive retry for partially-downloaded corpora.
set -u
BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download_phase3.log
mkdir -p "$BASE"
: > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

# --- CREMA-D: LFS pull -------------------------------------------------------
crema_d_lfs() {
  log "=== CREMA-D LFS pull ==="
  local dir=$BASE/CREMA-D
  if [[ -z $(ls "$dir/AudioWAV" 2>/dev/null) ]]; then
    (cd "$dir" && git lfs pull) 2>&1 | tail -20 | tee -a "$LOG"
  else
    log "  AudioWAV already populated — skip"
  fi
  local n=$(ls "$dir/AudioWAV" 2>/dev/null | wc -l)
  log "  AudioWAV contains $n files" ; du -sh "$dir/AudioWAV" 2>&1 | tee -a "$LOG"
}

# --- MUStARD++ videos: resume GDrive folder ----------------------------------
mustard_resume() {
  log "=== MUStARD++ videos resume ==="
  local dir=$BASE/MUStARD_Plus_Plus/videos
  mkdir -p "$dir"
  local folder_id='1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW'
  local have=$(find "$dir" -name "*.mp4" 2>/dev/null | wc -l)
  log "  before: $have mp4 files"
  (cd "$dir" && gdown --folder --continue "https://drive.google.com/drive/folders/${folder_id}" -O .) 2>&1 | tail -30 | tee -a "$LOG"
  local after=$(find "$dir" -name "*.mp4" 2>/dev/null | wc -l)
  log "  after:  $after mp4 files"
  du -sh "$dir" 2>&1 | tee -a "$LOG"
}

main() {
  crema_d_lfs
  mustard_resume
  log "=== final sizes ==="
  du -sh "$BASE"/*/ 2>/dev/null | tee -a "$LOG"
}

main "$@"
