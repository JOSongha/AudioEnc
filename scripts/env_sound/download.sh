#!/usr/bin/env bash
# Environmental sound datasets — fully open-access, Zenodo / GitHub hosted.
# Scope: non-speech audio classification + captioning corpora that Stage 2
# Tier-3 sound eval (stage2_eval_plan.md §3.2) assumes.
#
# Targets:
#   - Clotho v2.1      — captioning, dev+eval+val, ~15 GB                     — Zenodo 4783391
#   - FSD50K           — 51k clips / 200 labels, ~24 GB (dev+eval audio+labels) — Zenodo 4060432
#   - ESC-50           —  2k clips / 50 classes, ~600 MB                      — GitHub karolpiczak/ESC-50
#   - MACS             —  3.9k clips / multi-annotator captions, ~2 GB        — Zenodo 5114771
#
# Skipped (require registration or YouTube):
#   AudioSet, AudioCaps, VGGSound, UrbanSound8K, TAU Urban scenes.

set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=${BASE:-/mnt/tmp/datasets/env_sound}
LOG=$BASE/download.log
mkdir -p "$BASE" && : > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

fetch() {  # fetch URL outfile
  local url=$1 out=$2
  [[ -s $out ]] && { log "  already $out — skip"; return; }
  log "  wget $url"
  wget --no-verbose -c --timeout=60 -O "$out" "$url" 2>&1 | tee -a "$LOG"
}

# --- Clotho v2.1 -------------------------------------------------------------
clotho() {
  log "=== Clotho v2.1 (captioning; dev+eval+val splits) ==="
  local dir=$BASE/Clotho ; mkdir -p "$dir"
  local root='https://zenodo.org/records/4783391/files'
  fetch "$root/clotho_audio_development.7z?download=1"  "$dir/development.7z"
  fetch "$root/clotho_audio_evaluation.7z?download=1"   "$dir/evaluation.7z"
  fetch "$root/clotho_audio_validation.7z?download=1"   "$dir/validation.7z"
  fetch "$root/clotho_captions_development.csv?download=1"  "$dir/captions_development.csv"
  fetch "$root/clotho_captions_evaluation.csv?download=1"   "$dir/captions_evaluation.csv"
  fetch "$root/clotho_captions_validation.csv?download=1"   "$dir/captions_validation.csv"
  log "  Clotho size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- FSD50K ------------------------------------------------------------------
fsd50k() {
  log "=== FSD50K (200-class sound event recognition, 51k clips) ==="
  local dir=$BASE/FSD50K ; mkdir -p "$dir"
  local root='https://zenodo.org/records/4060432/files'
  # Multi-part zips (FSD50K.dev_audio.zip is split into .z01..z05 + .zip).
  for f in FSD50K.dev_audio.z01 FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 \
           FSD50K.dev_audio.z04 FSD50K.dev_audio.z05 FSD50K.dev_audio.zip \
           FSD50K.eval_audio.z01 FSD50K.eval_audio.zip \
           FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip; do
    fetch "$root/$f?download=1" "$dir/$f"
  done
  log "  FSD50K size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- ESC-50 ------------------------------------------------------------------
esc50() {
  log "=== ESC-50 (50-class, 2k clips) ==="
  local dir=$BASE/ESC-50
  if [[ -d $dir/audio ]]; then log "  already present — skip"; return; fi
  git clone --depth=1 https://github.com/karolpiczak/ESC-50 "$dir" 2>&1 | tee -a "$LOG"
  log "  ESC-50 size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- MACS --------------------------------------------------------------------
macs() {
  log "=== MACS (multi-annotator captions, ~3.9k clips) ==="
  local dir=$BASE/MACS ; mkdir -p "$dir"
  local root='https://zenodo.org/records/5114771/files'
  fetch "$root/MACS_audio.tar.gz?download=1" "$dir/MACS_audio.tar.gz"
  fetch "$root/MACS.yaml?download=1"         "$dir/MACS.yaml"
  fetch "$root/MACS_competence.yaml?download=1" "$dir/MACS_competence.yaml"
  log "  MACS size:" ; du -sh "$dir" | tee -a "$LOG"
}

main() {
  log "target: $BASE"
  df -h "$BASE" | tail -1 | tee -a "$LOG"

  esc50       # smallest first
  macs
  clotho
  fsd50k      # largest last

  log "=== final sizes ==="
  du -sh "$BASE"/*/ 2>/dev/null | tee -a "$LOG"
}

main "$@"
