#!/usr/bin/env bash
# Download open-access emotion source corpora for Stage 2 (no EULA required).
# Gated corpora (IEMOCAP / MSP-Podcast / SAVEE) are left to manual EULA flow.
# OMG / MOSEI require corpus-specific pipelines (YouTube + CMU SDK) — deferred.

set -u   # intentionally no -e: we want one failure to not stop the others

BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download.log
mkdir -p "$BASE"
: > "$LOG"   # truncate prior log

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. RAVDESS speech (Zenodo) -----------------------------------------------
ravdess() {
  log "=== RAVDESS ==="
  local dir=$BASE/RAVDESS
  local url='https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip'
  mkdir -p "$dir"
  if [[ -d $dir/Actor_01 ]]; then log "  already unzipped — skip"; return; fi
  wget --no-verbose -c -O "$dir/speech.zip" "$url" 2>&1 | tee -a "$LOG"
  (cd "$dir" && unzip -q -n speech.zip) 2>&1 | tee -a "$LOG"
  log "  unzipped to $dir"
}

# --- 2. MUStARD++ (GitHub) ----------------------------------------------------
mustard() {
  log "=== MUStARD++ ==="
  local dir=$BASE/MUStARD_Plus_Plus
  if [[ -d $dir/.git ]]; then log "  already cloned — skip"; return; fi
  git clone --depth=1 https://github.com/cfiltnlp/MUStARD_Plus_Plus "$dir" 2>&1 | tee -a "$LOG"
  log "  NOTE: MUStARD++ repo ships download URLs / CSVs; audio may need a second"
  log "        step via MUStARD_Plus_Plus/final_mustard++.csv and youtube-dl."
}

# --- 3. CREMA-D (GitHub + git-lfs required) -----------------------------------
crema_d() {
  log "=== CREMA-D ==="
  local dir=$BASE/CREMA-D
  if [[ -d $dir/AudioWAV ]]; then log "  already present — skip"; return; fi
  if ! have git-lfs; then
    log "  SKIP: git-lfs not installed. Install git-lfs then: "
    log "        git lfs install && git clone https://github.com/CheyneyComputerScience/CREMA-D $dir"
    return
  fi
  git lfs install 2>&1 | tee -a "$LOG"
  git clone https://github.com/CheyneyComputerScience/CREMA-D "$dir" 2>&1 | tee -a "$LOG"
}

# --- 4. TESS (U. Toronto TSpace) ----------------------------------------------
tess() {
  log "=== TESS ==="
  local dir=$BASE/TESS
  mkdir -p "$dir"
  if [[ -n $(ls "$dir"/*.wav 2>/dev/null) ]]; then log "  already present — skip"; return; fi
  log "  NOTE: TESS on TSpace (handle 1807/24487) has no single zip endpoint;"
  log "        files are per-emotion folder zips. Manual steps:"
  log "          1. Browse https://tspace.library.utoronto.ca/handle/1807/24487"
  log "          2. Download each per-emotion zip into $dir/"
  log "          3. unzip them flat into $dir/"
  log "        Alternative: HuggingFace mirror 'Ar4ikov/iemocap_ravdess_tess' or similar"
  log "        — verify license / integrity before use."
}

# --- 5. ESD (HLT Singapore, GDrive) -------------------------------------------
esd() {
  log "=== ESD (Emotion Speech Dataset) ==="
  local dir=$BASE/ESD
  mkdir -p "$dir"
  if [[ -n $(ls "$dir"/0011 2>/dev/null) ]]; then log "  already present — skip"; return; fi
  log "  NOTE: ESD is distributed via Google Drive from:"
  log "        https://github.com/HLTSingapore/Emotional-Speech-Data"
  log "        Requires 'gdown' (pip install gdown) or manual browser download."
  log "        GDrive folder link is in the repo README — expect ~2 GB zip."
}

# --- 6. MELD (raw tar) --------------------------------------------------------
meld() {
  log "=== MELD ==="
  local dir=$BASE/MELD
  mkdir -p "$dir"
  local url='http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
  if [[ -d $dir/train_splits ]]; then log "  already extracted — skip"; return; fi
  log "  trying $url"
  if wget --no-verbose -c --timeout=30 -O "$dir/MELD.Raw.tar.gz" "$url" 2>&1 | tee -a "$LOG"; then
    (cd "$dir" && tar -xzf MELD.Raw.tar.gz) 2>&1 | tee -a "$LOG"
  else
    log "  SKIP: download failed. Alt: github.com/declare-lab/MELD README lists mirror."
  fi
  log "  POST: MELD raw is video (.mp4) — audio extraction needed (ffmpeg -i X.mp4 X.wav)."
}

# --- 7/8. OMG-Emotion, CMU-MOSEI — deferred (pipelines) -----------------------
deferred_notes() {
  log "=== OMG-Emotion (DEFERRED) ==="
  log "  YouTube-based. Use github.com/knowledgetechnologyuhh/OMGEmotionChallenge"
  log "  with yt-dlp; expect partial yield (dead links)."
  log "=== CMU-MOSEI (DEFERRED) ==="
  log "  Use CMU-MultimodalSDK: github.com/A2Zadeh/CMU-MultimodalSDK"
  log "  Requires Python pipeline + ~65 GB disk."
}

main() {
  log "target base: $BASE"
  log "free:" ; df -h "$BASE" | tail -1 | tee -a "$LOG"

  ravdess
  mustard
  crema_d
  tess
  esd
  meld
  deferred_notes

  log "=== done. Summary: ==="
  (cd "$BASE" && du -sh * 2>/dev/null | tee -a "$LOG")
}

main "$@"
