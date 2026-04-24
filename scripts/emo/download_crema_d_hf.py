"""Try to obtain CREMA-D audio via HuggingFace, since GitHub LFS quota is
exhausted (phase-3 log shows 'This repository exceeded its LFS budget').

Strategy: search HF Hub for 'crema', try each candidate with `load_dataset`,
keep the first one that exposes audio + labels, dump WAVs out into a flat
directory so downstream matching against `CREMA-D_1001_IEO_ANG_LO` IDs is
straightforward.
"""
from __future__ import annotations

import io
import shutil
import sys
import traceback
from pathlib import Path

OUT = Path("/mnt/tmp/datasets/emotion_raw/CREMA-D/AudioWAV")
OUT.mkdir(parents=True, exist_ok=True)


def try_dataset(repo_id: str) -> bool:
    """Attempt to load a HF dataset mirror and extract WAVs."""
    print(f"\n== try {repo_id} ==", flush=True)
    try:
        from datasets import load_dataset  # import here so failure isolates
    except Exception as e:
        print(f"   datasets import failed: {e}")
        return False
    try:
        ds = load_dataset(repo_id, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"   load_dataset failed: {type(e).__name__}: {e}")
        return False
    cols = ds.column_names
    print(f"   columns: {cols}")
    # try to find audio + filename column
    audio_col = next((c for c in cols if c in ("audio", "wav", "speech")), None)
    name_col = next((c for c in cols if c in ("file", "filename", "path", "id")), None)
    if audio_col is None:
        print("   no audio column — skip")
        return False
    import soundfile as sf
    n = 0
    for row in ds:
        audio = row[audio_col]
        name = row.get(name_col, None) if name_col else None
        if name is None or not isinstance(name, str):
            # fall back to a numeric id
            name = f"crema_{n:05d}.wav"
        stem = Path(name).stem
        if not stem.endswith(".wav"):
            stem = f"{stem}.wav"
        out = OUT / Path(stem).name
        if out.exists():
            n += 1
            continue
        if isinstance(audio, dict) and "array" in audio:
            sf.write(out, audio["array"], audio["sampling_rate"])
        elif isinstance(audio, dict) and "bytes" in audio:
            out.write_bytes(audio["bytes"])
        elif isinstance(audio, dict) and "path" in audio and audio["path"]:
            shutil.copy(audio["path"], out)
        else:
            print(f"   unknown audio payload: {type(audio)}")
            return False
        n += 1
        if n % 500 == 0:
            print(f"   wrote {n} files", flush=True)
    print(f"   done: wrote {n} files to {OUT}")
    return n > 0


def main() -> int:
    candidates = [
        "AbstractTNT/Crema-D",
        "AbstractTNT/CREMA-D",
        "silpakanneganti/cremad",
        "confit/crema-d",
        "mteb/crema_d",
        "ajyy/CREMA_D",
        "Ar4ikov/iemocap_ravdess_crema",
    ]
    if len(sys.argv) > 1:
        candidates = sys.argv[1:] + candidates
    for c in candidates:
        try:
            if try_dataset(c):
                print(f"\nSUCCESS via {c}")
                return 0
        except Exception:
            traceback.print_exc()
    print("\nAll candidates failed. Options:")
    print("  1. Kaggle: ejlok1/cremad — needs `kaggle` CLI + API token")
    print("  2. Contact repo maintainer to increase LFS budget")
    print("  3. Download CREMA-D MP3 (smaller; AudioMP3 dir) from a different mirror")
    return 1


if __name__ == "__main__":
    sys.exit(main())
