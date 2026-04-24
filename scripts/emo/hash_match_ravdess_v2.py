"""RAVDESS hash match v2 — resample-normalized.

v1 (hash_match_ravdess.py) hashed raw WAV bytes and matched 0/200 because
LISTEN re-encoded the audio (different sample rate / bit depth) before
storing. This version resamples both sides to a canonical 16 kHz mono int16
representation, then hashes the first 4 096 samples.

Match rate above ~60% validates the approach; lower than that probably means
LISTEN did content-level transforms (denoise/normalize) beyond resampling.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sps

LISTEN_PARQUET = Path("/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet")
RAW = Path("/mnt/tmp/datasets/emotion_raw/RAVDESS")
OUT = Path("/mnt/tmp/listen_analysis/exclude_manifests/ravdess_hash_match.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

CANON_SR = 16000
HEAD_SAMPLES = 4096


def canonicalize(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono int16; return head slice."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != CANON_SR:
        n_out = int(round(len(wav) * CANON_SR / sr))
        wav = sps.resample(wav, n_out)
    # to int16 (matches LISTEN's WAV encoding in parquet bytes)
    wav = np.clip(wav * 32767.0 if wav.dtype.kind == "f" else wav, -32768, 32767).astype(np.int16)
    head = wav[:HEAD_SAMPLES]
    # pad to fixed length so hash is deterministic
    if len(head) < HEAD_SAMPLES:
        head = np.concatenate([head, np.zeros(HEAD_SAMPLES - len(head), dtype=np.int16)])
    return head


def key_of(head: np.ndarray, nsamples: int) -> tuple:
    # include length-bucket (100-ms res) so near-duplicates don't collide
    bucket = nsamples // (CANON_SR // 10)
    return (bucket, hashlib.md5(head.tobytes()).hexdigest())


def main() -> None:
    wavs = sorted(RAW.rglob("*.wav"))
    print(f"indexing {len(wavs)} raw RAVDESS wavs at canonical 16 kHz int16…")
    raw_index: dict[tuple, str] = {}
    for p in wavs:
        data, sr = sf.read(str(p))
        head = canonicalize(np.asarray(data, dtype=np.float64), sr)
        # length after canonicalization
        n = len(data) if data.ndim == 1 else len(data)
        n_canon = int(round(n * CANON_SR / sr))
        raw_index[key_of(head, n_canon)] = str(p)
    print(f"  indexed {len(raw_index)} unique keys")

    test = pd.read_parquet(LISTEN_PARQUET)
    rav = test[test["dataset_source"] == "RAVDESS"]
    print(f"matching {len(rav)} LISTEN-test RAVDESS rows…")

    mapped, unmapped = {}, []
    for _, row in rav.iterrows():
        b = row["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(b))
        head = canonicalize(np.asarray(data, dtype=np.float64), sr)
        n = len(data) if data.ndim == 1 else len(data)
        n_canon = int(round(n * CANON_SR / sr))
        k = key_of(head, n_canon)
        path = raw_index.get(k)
        if path:
            mapped[row["id"]] = path
        else:
            unmapped.append(row["id"])

    print(f"  mapped   {len(mapped)}  ({len(mapped) / len(rav) * 100:.1f}%)")
    print(f"  unmapped {len(unmapped)}")
    OUT.write_text(json.dumps({"mapped": mapped, "unmapped": unmapped}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
