"""RAVDESS hash-match: map LISTEN-test RAVDESS ids (sequential) to raw Actor wav files.

LISTEN IDs use `RAVDESS_train_{idx}_{emo}_{intensity}` which is not directly
convertible to the RAVDESS filename (`03-01-EE-II-SS-RR-AA.wav`). The only
deterministic mapping is to match by audio content.

Strategy: compute (samplerate, frame-count, first-N-samples-hash) for each raw
RAVDESS WAV, then for each LISTEN-test RAVDESS row decode its audio bytes and
look up the same key. Collision-free because raw WAVs are PCM at matching SR.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

LISTEN_PARQUET = Path("/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet")
RAW = Path("/mnt/tmp/datasets/emotion_raw/RAVDESS")
OUT = Path("/mnt/tmp/listen_analysis/exclude_manifests/ravdess_hash_match.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def key_of(arr, sr: int) -> tuple:
    n = len(arr)
    head = arr[:4096].tobytes() if n >= 4096 else arr.tobytes()
    h = hashlib.md5(head).hexdigest()
    return (sr, n, h)


def main() -> None:
    raw_index: dict[tuple, str] = {}
    wavs = sorted(RAW.rglob("*.wav"))
    print(f"indexing {len(wavs)} raw RAVDESS wavs…")
    for p in wavs:
        data, sr = sf.read(str(p), dtype="int16")
        if data.ndim > 1:
            data = data[:, 0]
        raw_index[key_of(data, sr)] = str(p)
    print(f"  indexed {len(raw_index)} unique keys")

    test = pd.read_parquet(LISTEN_PARQUET)
    ravdess = test[test["dataset_source"] == "RAVDESS"]
    print(f"matching {len(ravdess)} LISTEN-test RAVDESS rows…")

    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    for _, row in ravdess.iterrows():
        audio_bytes = row["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        if data.ndim > 1:
            data = data[:, 0]
        k = key_of(data, sr)
        path = raw_index.get(k)
        if path:
            mapped[row["id"]] = path
        else:
            unmapped.append(row["id"])

    print(f"  mapped   {len(mapped)}")
    print(f"  unmapped {len(unmapped)}")
    OUT.write_text(json.dumps({"mapped": mapped, "unmapped": unmapped}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
