#!/usr/bin/env bash
# Phase 2: downloads that became possible after installing git-lfs + gdown via mamba.
# Also extracts nested MELD tarballs that phase 1 left packed.

set -u
BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download_phase2.log
mkdir -p "$BASE"
: > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

# --- MELD nested tar extraction ----------------------------------------------
meld_extract() {
  log "=== MELD nested extraction ==="
  local raw=$BASE/MELD/MELD.Raw
  for split in train dev test; do
    local tarball=$raw/${split}.tar.gz
    local outdir=$raw/${split}_splits
    if [[ -d $outdir ]]; then log "  $split already extracted — skip"; continue; fi
    [[ -f $tarball ]] || { log "  $tarball missing — skip"; continue; }
    log "  extracting $tarball"
    (cd "$raw" && tar -xzf ${split}.tar.gz) 2>&1 | tee -a "$LOG"
  done
  log "  MELD.Raw dir:" ; du -sh "$raw"/* | tee -a "$LOG"
}

# --- CREMA-D via git-lfs -----------------------------------------------------
crema_d() {
  log "=== CREMA-D (git-lfs) ==="
  local dir=$BASE/CREMA-D
  if [[ -d $dir/AudioWAV ]]; then log "  already present — skip"; return; fi
  rm -rf "$dir"
  git lfs install --skip-repo 2>&1 | tee -a "$LOG"
  git clone https://github.com/CheyneyComputerScience/CREMA-D "$dir" 2>&1 | tee -a "$LOG"
  log "  CREMA-D size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- MUStARD++ videos via gdown (public GDrive folder) -----------------------
mustard_videos() {
  log "=== MUStARD++ videos (gdown folder) ==="
  local dir=$BASE/MUStARD_Plus_Plus/videos
  mkdir -p "$dir"
  local folder_id='1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW'
  if [[ -n $(ls "$dir" 2>/dev/null) ]]; then log "  already populated — skip"; return; fi
  (cd "$dir" && gdown --folder "https://drive.google.com/drive/folders/${folder_id}" -O .) 2>&1 | tail -20 | tee -a "$LOG"
  log "  MUStARD++ videos size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- TESS — attempt direct-link scrape of TSpace ------------------------------
tess() {
  log "=== TESS (TSpace per-folder download) ==="
  local dir=$BASE/TESS
  mkdir -p "$dir"
  if [[ -n $(ls "$dir"/*.wav 2>/dev/null) ]]; then log "  already present — skip"; return; fi
  log "  TSpace does not expose a programmatic ZIP endpoint reliably."
  log "  Skipping automated path. Manual options:"
  log "    a) browse handle 1807/24487, download per-emotion zips"
  log "    b) HuggingFace mirror (verify license)"
  log "  TESS is 1.6 h — low priority. Skip for now."
}

main() {
  log "target: $BASE"
  meld_extract
  crema_d
  mustard_videos
  tess

  log "=== final sizes ==="
  du -sh "$BASE"/*/ 2>/dev/null | tee -a "$LOG"
}

main "$@"
