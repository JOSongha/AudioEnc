"""
Download RAVDESS + CREMA-D from HuggingFace mirrors.

Output layout:
  /mnt/tmp/datasets/emotion_raw/CREMA-D/AudioWAV/<utt_id>.wav
  /mnt/tmp/datasets/emotion_raw/RAVDESS/AudioSpeech/<utt_id>.wav

CREMA-D filename: 1001_IEO_ANG_LO.wav  (actor_sentence_emotion_intensity)
RAVDESS filename: 03-01-06-02-01-02-15.wav  (modality-channel-emotion-intensity-statement-repetition-actor)
"""

import io
import shutil
import sys
import traceback
from pathlib import Path

OUT_ROOT = Path("/mnt/tmp/datasets/emotion_raw")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── CREMA-D ──────────────────────────────────────────────────────────────────

CREMAD_OUT = OUT_ROOT / "CREMA-D/AudioWAV"
CREMAD_CANDIDATES = [
    "AbstractTNT/Crema-D",
    "silpakanneganti/cremad",
    "confit/crema-d",
    "mteb/crema_d",
    "ajyy/CREMA_D",
]


def write_wav(audio, path: Path) -> bool:
    import soundfile as sf
    try:
        if isinstance(audio, dict) and "array" in audio:
            sf.write(path, audio["array"], audio["sampling_rate"])
            return True
        if isinstance(audio, dict) and "bytes" in audio and audio["bytes"]:
            path.write_bytes(audio["bytes"])
            return True
        if isinstance(audio, dict) and "path" in audio and audio["path"]:
            shutil.copy(audio["path"], path)
            return True
    except Exception as e:
        print(f"  write fail: {e}")
    return False


def download_cremad_via_parquet():
    """
    Download parquet files directly + decode audio bytes with soundfile.
    Bypasses datasets library which needs torchcodec/ffmpeg.
    """
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    CREMAD_OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    for repo_id in CREMAD_CANDIDATES:
        print(f"\n== CREMA-D parquet: try {repo_id} ==", flush=True)
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception as e:
            print(f"   list_repo failed: {type(e).__name__}: {e}")
            continue
        parquets = [f for f in files if f.endswith(".parquet")]
        if not parquets:
            print(f"   no parquet files; available: {files[:5]}")
            continue
        print(f"   found {len(parquets)} parquet files")

        n_total = 0
        for pq_file in parquets:
            try:
                local = hf_hub_download(repo_id, pq_file, repo_type="dataset")
            except Exception as e:
                print(f"   download {pq_file} failed: {e}")
                continue
            tbl = pq.read_table(local)
            cols = tbl.column_names
            audio_col = next((c for c in cols if c in ("audio", "wav", "speech")), None)
            name_col = next((c for c in cols if c in ("file", "filename", "path", "id")), None)
            if audio_col is None:
                print(f"   no audio col in {pq_file}: {cols}")
                continue
            audio_data = tbl[audio_col].to_pylist()
            names = tbl[name_col].to_pylist() if name_col else [None] * len(audio_data)

            for i, (audio, name) in enumerate(zip(audio_data, names)):
                if not (isinstance(name, str) and name):
                    name = f"crema_{n_total:05d}.wav"
                stem = Path(name).stem
                out = CREMAD_OUT / f"{stem}.wav"
                if out.exists():
                    n_total += 1
                    continue
                # audio is dict with 'bytes' key (for parquet HF audio)
                try:
                    if isinstance(audio, dict) and audio.get("bytes"):
                        out.write_bytes(audio["bytes"])
                        n_total += 1
                        if n_total % 500 == 0:
                            print(f"   wrote {n_total}")
                    elif isinstance(audio, dict) and "array" in audio and audio["array"] is not None:
                        sf.write(out, audio["array"], audio["sampling_rate"])
                        n_total += 1
                except Exception as e:
                    print(f"   write {stem} failed: {e}")

        print(f"   done: {n_total} files written")
        if len(list(CREMAD_OUT.glob("*.wav"))) >= 1000:
            return True

    print("ALL CREMA-D parquet mirrors failed")
    return False


def download_cremad():
    CREMAD_OUT.mkdir(parents=True, exist_ok=True)
    if len(list(CREMAD_OUT.glob("*.wav"))) >= 7000:
        print(f"CREMA-D already has {len(list(CREMAD_OUT.glob('*.wav')))} files — skip")
        return True
    return download_cremad_via_parquet()


# ── RAVDESS ──────────────────────────────────────────────────────────────────

RAVDESS_OUT = OUT_ROOT / "RAVDESS/AudioSpeech"
RAVDESS_CANDIDATES = [
    "narad/ravdess",
    "confit/ravdess",
    "mehulgawri/RAVDESS",
    "ajyy/RAVDESS",
    "Ar4ikov/iemocap_ravdess_audio_emotion_classification",  # combined; filter later
]


def download_ravdess():
    RAVDESS_OUT.mkdir(parents=True, exist_ok=True)
    if len(list(RAVDESS_OUT.glob("*.wav"))) >= 1000:
        print(f"RAVDESS already has {len(list(RAVDESS_OUT.glob('*.wav')))} files — skip")
        return True
    from datasets import load_dataset

    for repo_id in RAVDESS_CANDIDATES:
        print(f"\n== RAVDESS: try {repo_id} ==", flush=True)
        try:
            ds = load_dataset(repo_id, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"   load failed: {type(e).__name__}: {e}")
            continue
        cols = ds.column_names
        print(f"   columns: {cols}")
        audio_col = next((c for c in cols if c in ("audio", "wav", "speech")), None)
        if audio_col is None:
            print("   no audio col — skip")
            continue
        name_col = next((c for c in cols if c in ("file", "filename", "path", "id")), None)
        n = 0
        for i, row in enumerate(ds):
            audio = row[audio_col]
            name = row.get(name_col) if name_col else None
            if not (isinstance(name, str) and name):
                name = f"ravdess_{i:05d}.wav"
            stem = Path(name).stem
            out = RAVDESS_OUT / f"{stem}.wav"
            if out.exists():
                n += 1
                continue
            if write_wav(audio, out):
                n += 1
                if n % 200 == 0:
                    print(f"   wrote {n}")
        print(f"   done: {n} files")
        if len(list(RAVDESS_OUT.glob("*.wav"))) >= 500:
            return True
    print("ALL RAVDESS mirrors failed")
    return False


# ── Zenodo fallback for RAVDESS ──────────────────────────────────────────────

def download_ravdess_zenodo():
    """Direct Zenodo download if HF mirrors fail."""
    import urllib.request
    import zipfile

    out_zip = OUT_ROOT / "RAVDESS/Audio_Speech_Actors_01-24.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if not out_zip.exists():
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, out_zip)

    extract_dir = OUT_ROOT / "RAVDESS/zenodo_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip) as zf:
        zf.extractall(extract_dir)

    # Move all .wav into AudioSpeech/
    n = 0
    for wav in extract_dir.rglob("*.wav"):
        dst = RAVDESS_OUT / wav.name
        if not dst.exists():
            shutil.copy(wav, dst)
            n += 1
    print(f"Zenodo RAVDESS: {n} new wavs")
    return n > 0


def main():
    print("=== CREMA-D ===")
    cd_ok = False
    try:
        cd_ok = download_cremad()
    except Exception as e:
        print(f"CREMA-D exception: {e}")
        traceback.print_exc()

    print("\n=== RAVDESS ===")
    rv_ok = False
    try:
        rv_ok = download_ravdess()
    except Exception as e:
        print(f"RAVDESS exception: {e}")
        traceback.print_exc()

    if not rv_ok:
        print("HF RAVDESS failed — try Zenodo direct...")
        try:
            rv_ok = download_ravdess_zenodo()
        except Exception as e:
            print(f"Zenodo RAVDESS failed: {e}")

    print(f"\n=== Summary ===")
    print(f"CREMA-D: {len(list(CREMAD_OUT.glob('*.wav')))} wavs at {CREMAD_OUT}")
    print(f"RAVDESS: {len(list(RAVDESS_OUT.glob('*.wav')))} wavs at {RAVDESS_OUT}")


if __name__ == "__main__":
    main()
