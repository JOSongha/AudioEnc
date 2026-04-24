#!/usr/bin/env bash
# Extract Clotho (.7z) and FSD50K (multi-part zip) into flat audio dirs.
set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=${BASE:-/mnt/tmp/datasets/env_sound}
LOG=$BASE/extract.log
: > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

# --- Clotho 7z -----------------------------------------------------
clotho() {
  log "=== Clotho extraction ==="
  local dir=$BASE/Clotho
  for s in development evaluation validation; do
    if [[ -d $dir/$s ]]; then log "  $s already extracted"; continue; fi
    [[ -f $dir/$s.7z ]] || { log "  missing $s.7z — skip"; continue; }
    (cd "$dir" && 7z x -y $s.7z) >>"$LOG" 2>&1
    local n=$(find "$dir/$s" -name "*.wav" 2>/dev/null | wc -l)
    log "  $s: $n wavs"
  done
  du -sh "$dir" | tee -a "$LOG"
}

# --- FSD50K multi-part zip join + unzip ------------------------------------
fsd50k() {
  log "=== FSD50K extraction ==="
  local dir=$BASE/FSD50K
  # dev audio: z01-z05 + zip (main). Concatenate + unzip.
  if [[ ! -d $dir/FSD50K.dev_audio ]]; then
    if [[ -f $dir/FSD50K.dev_audio.zip && -f $dir/FSD50K.dev_audio.z01 ]]; then
      log "  joining FSD50K.dev_audio split…"
      (cd "$dir" && zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio_joined.zip) >>"$LOG" 2>&1
      (cd "$dir" && unzip -q -n FSD50K.dev_audio_joined.zip) >>"$LOG" 2>&1
      rm -f "$dir/FSD50K.dev_audio_joined.zip"
    fi
  fi
  # eval audio: z01 + zip
  if [[ ! -d $dir/FSD50K.eval_audio ]]; then
    if [[ -f $dir/FSD50K.eval_audio.zip && -f $dir/FSD50K.eval_audio.z01 ]]; then
      log "  joining FSD50K.eval_audio split…"
      (cd "$dir" && zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio_joined.zip) >>"$LOG" 2>&1
      (cd "$dir" && unzip -q -n FSD50K.eval_audio_joined.zip) >>"$LOG" 2>&1
      rm -f "$dir/FSD50K.eval_audio_joined.zip"
    fi
  fi
  # ground truth + metadata + doc: plain zips
  for z in FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip; do
    [[ -f $dir/$z ]] || continue
    (cd "$dir" && unzip -q -n "$z") >>"$LOG" 2>&1
  done
  log "  FSD50K dev:  $(find "$dir/FSD50K.dev_audio" -name '*.wav' 2>/dev/null | wc -l) wavs"
  log "  FSD50K eval: $(find "$dir/FSD50K.eval_audio" -name '*.wav' 2>/dev/null | wc -l) wavs"
  du -sh "$dir" | tee -a "$LOG"
}

main() {
  clotho
  fsd50k
  log "=== final ==="; du -sh "$BASE"/*/ | tee -a "$LOG"
}

main "$@"
