#!/usr/bin/env bash
# English, audio-only (no video), fully open-access emotion datasets not already
# in the LISTEN source list. Targets:
#   - DailyTalk  (23 k turns, dialogue, EN, emotion labels) — Zenodo
#   - EmoV-DB    ( 3 EN speakers × 5 emotions, ~7 k utt)   — GitHub / OneDrive mirrors
# Falls back to cloning the upstream repo so the README's current URL is
# available in situ if the hard-coded URL here becomes stale.

set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download_english_emo.log
mkdir -p "$BASE" && : > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

# --- DailyTalk ---------------------------------------------------------------
dailytalk() {
  log "=== DailyTalk (dialogue, EN, 7 emotions, 23k turns) ==="
  local dir=$BASE/DailyTalk
  mkdir -p "$dir"
  if [[ -d $dir/dataset ]]; then log "  already present — skip"; return; fi
  # clone repo (small — has metadata + reference to audio hosting)
  [[ -d $dir/repo ]] || git clone --depth=1 https://github.com/keonlee9420/DailyTalk "$dir/repo" 2>&1 | tee -a "$LOG"
  log "  README excerpt:"
  grep -iE "download|zenodo|drive|dataset\.tar|hf\.co|huggingface" "$dir/repo/README.md" 2>/dev/null \
    | head -15 | sed 's/^/    /' | tee -a "$LOG"
  log "  ---"
  log "  trying known Zenodo mirror (8025726)…"
  wget --no-verbose -c --timeout=60 -O "$dir/dailytalk.tar.gz" \
    "https://zenodo.org/records/8025726/files/dataset.tar.gz" 2>&1 | tee -a "$LOG"
  if [[ -s $dir/dailytalk.tar.gz ]]; then
    (cd "$dir" && tar -xzf dailytalk.tar.gz) 2>&1 | tee -a "$LOG"
    log "  DailyTalk size:" ; du -sh "$dir" | tee -a "$LOG"
  else
    log "  zenodo 8025726 URL failed — check $dir/repo/README.md for current URL"
  fi
}

# --- EmoV-DB -----------------------------------------------------------------
emov_db() {
  log "=== EmoV-DB (3 EN speakers × 5 emotions, ~7k utt) ==="
  local dir=$BASE/EmoV-DB
  mkdir -p "$dir"
  if [[ -n $(ls "$dir"/*.wav 2>/dev/null) ]] || [[ -d $dir/bea ]]; then
    log "  already present — skip"; return
  fi
  [[ -d $dir/repo ]] || git clone --depth=1 https://github.com/numediart/EmoV-DB "$dir/repo" 2>&1 | tee -a "$LOG"
  log "  README excerpt:"
  grep -iE "download|onedrive|drive|dataset|hf\.co|huggingface|zenodo" "$dir/repo/README.md" 2>/dev/null \
    | head -20 | sed 's/^/    /' | tee -a "$LOG"
  log "  ---"
  # EmoV-DB maintainers distribute via OneDrive / Google Drive links that rotate.
  # Try HF mirror first (some community reuploads exist).
  for candidate in "confit/emov-db" "confit/EmoV_DB" "Ar4ikov/EmoV-DB"; do
    log "  HF candidate: $candidate"
    python3 - <<PY 2>&1 | tee -a "$LOG"
try:
    from datasets import load_dataset
    ds = load_dataset("$candidate")
    import os, soundfile as sf
    out = "$dir/audio"; os.makedirs(out, exist_ok=True)
    n = 0
    for split in ds:
        for row in ds[split]:
            audio = row.get("audio") or row.get("wav")
            if audio is None: continue
            name = str(row.get("file") or row.get("id") or f"emov_{n:05d}")
            if not name.endswith(".wav"): name += ".wav"
            path = os.path.join(out, os.path.basename(name))
            if isinstance(audio, dict) and "array" in audio:
                sf.write(path, audio["array"], audio["sampling_rate"])
            n += 1
    print(f"EMOV OK via $candidate: wrote {n} wavs")
except Exception as e:
    print(f"EMOV fail $candidate: {type(e).__name__}: {e}")
PY
    if [[ -n $(ls "$dir/audio" 2>/dev/null) ]]; then log "  success via $candidate"; return; fi
  done
  log "  HF mirrors all failed — manual OneDrive step required (see README)."
}

main() {
  log "target: $BASE"
  df -h "$BASE" | tail -1 | tee -a "$LOG"

  dailytalk
  emov_db

  log "=== sizes ==="
  du -sh "$BASE"/*/ 2>/dev/null | tee -a "$LOG"
}

main "$@"
