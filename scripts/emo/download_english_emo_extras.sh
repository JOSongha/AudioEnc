#!/usr/bin/env bash
# Additional English audio-text emotion datasets (round 3).
# Targets:
#   - SEND         Stanford Emotional Narratives (~28h spontaneous narrative)
#   - JL-Corpus    NZ English acted (~2.4k utt, 2 speakers)
#   - EMNS         Emotional Multi-speaker Narrative Set (~1.2k utt, TTS-quality)
#   - MSP-IMPROV / MSP-Conversation: EULA-pending, not downloaded here.
set -u
export PATH=/mnt/ddn/users/jos/miniforge3/bin:$PATH

BASE=${BASE:-/mnt/tmp/datasets/emotion_raw}
LOG=$BASE/download_english_emo_extras.log
: > "$LOG"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"; }

# --- SEND --------------------------------------------------------------------
send_corpus() {
  log "=== SEND (Stanford Emotional Narratives) ==="
  local dir=$BASE/SEND
  mkdir -p "$dir"
  if [[ -d $dir/repo ]]; then log "  already cloned — skip"
  else
    git clone --depth=1 https://github.com/StanfordSocialNeuroscienceLab/SEND "$dir/repo" 2>&1 | tee -a "$LOG"
  fi
  log "  README excerpt (download instructions):"
  grep -iE "download|drive|dropbox|zenodo|osf|audio|stanford|http" "$dir/repo/README.md" 2>/dev/null \
    | head -15 | sed 's/^/    /' | tee -a "$LOG"
  log "  ---"
  log "  Checking for LFS blobs or data dirs…"
  (cd "$dir/repo" && git lfs ls-files 2>&1 | head -10) | tee -a "$LOG"
  find "$dir/repo" -maxdepth 3 -type d | tee -a "$LOG"
  log "  SEND initial size:" ; du -sh "$dir" | tee -a "$LOG"
}

# --- JL-Corpus ---------------------------------------------------------------
jl_corpus() {
  log "=== JL-Corpus (NZ English, 2 speakers) ==="
  local dir=$BASE/JL-Corpus
  mkdir -p "$dir"
  # Kaggle path requires creds; try direct (kaggle.com is HTTP-locked without auth).
  # Fallback candidates: author's Mendeley / GitHub hosting / HF mirror.
  # Author's mirror (Jesin James): https://www.kaggle.com/datasets/tli725/jl-corpus
  # HF mirror attempts:
  for candidate in "confit/JL-Corpus" "Ar4ikov/JL-Corpus" "SER-DATASETS/JL-Corpus"; do
    log "  HF: $candidate"
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
            name = str(row.get("file") or row.get("id") or row.get("path") or f"jl_{n:05d}")
            if audio is None: continue
            if not name.endswith(".wav"): name += ".wav"
            p = os.path.join(out, os.path.basename(name))
            if isinstance(audio, dict) and "array" in audio:
                sf.write(p, audio["array"], audio["sampling_rate"])
            n += 1
    print(f"JL OK via $candidate: {n} wavs")
except Exception as e:
    print(f"JL fail $candidate: {type(e).__name__}: {e}")
PY
    if [[ -n $(ls "$dir/audio" 2>/dev/null) ]]; then log "  success via $candidate"; return; fi
  done
  log "  All HF candidates failed — Kaggle CLI + API token required:"
  log "    pip install --user kaggle && echo '{user,key}' > ~/.kaggle/kaggle.json"
  log "    kaggle datasets download tli725/jl-corpus -p $dir && unzip $dir/jl-corpus.zip -d $dir"
}

# --- EMNS --------------------------------------------------------------------
emns() {
  log "=== EMNS (Emotional Multi-speaker Narrative Set) ==="
  local dir=$BASE/EMNS
  mkdir -p "$dir"
  if [[ -d $dir/repo ]]; then log "  already cloned — skip"
  else
    git clone --depth=1 https://github.com/knoriy/EMNS-Dataset "$dir/repo" 2>&1 | tee -a "$LOG"
  fi
  log "  README excerpt:"
  grep -iE "download|huggingface|hf\.co|drive|zenodo|\.zip|\.tar" "$dir/repo/README.md" 2>/dev/null \
    | head -15 | sed 's/^/    /' | tee -a "$LOG"
  log "  ---"
  # Try HF direct (EMNS is hosted on HF according to most references)
  for candidate in "knoriy/EMNS" "knoriy/EMNS-Dataset"; do
    log "  HF: $candidate"
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
            name = str(row.get("file") or row.get("utterance_id") or row.get("id") or f"emns_{n:05d}")
            if audio is None: continue
            if not name.endswith(".wav"): name += ".wav"
            p = os.path.join(out, os.path.basename(name))
            if isinstance(audio, dict) and "array" in audio:
                sf.write(p, audio["array"], audio["sampling_rate"])
            n += 1
    print(f"EMNS OK via $candidate: {n} wavs")
except Exception as e:
    print(f"EMNS fail $candidate: {type(e).__name__}: {e}")
PY
    if [[ -n $(ls "$dir/audio" 2>/dev/null) ]]; then log "  success via $candidate"; return; fi
  done
  log "  HF candidates failed — check $dir/repo/README.md for release URL"
}

main() {
  log "target: $BASE"

  # run all three concurrently — each writes into a distinct subdir
  send_corpus    &
  jl_corpus      &
  emns           &
  wait

  log "=== final sizes ==="
  du -sh "$BASE/SEND" "$BASE/JL-Corpus" "$BASE/EMNS" 2>/dev/null | tee -a "$LOG"
}

main "$@"
