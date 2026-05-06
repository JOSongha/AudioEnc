"""
Download NSynth test (full 4096) + train subset (15 shards = ~30k) from
confit/nsynth-parquet (HF mirror), decode audio bytes to wav files, build manifest.
"""

import io
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("HF_HOME", "/mnt/tmp/cache/huggingface")
from huggingface_hub import hf_hub_download

REPO = "confit/nsynth-parquet"
OUT_ROOT = Path("/mnt/tmp/datasets/music/nsynth")
TEST_DIR = OUT_ROOT / "test"
TRAIN_DIR = OUT_ROOT / "train_30k"
TEST_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_SHARDS = 15  # ~15 × 2048 = ~30,720 samples


def save_wav_bytes(audio_dict, out_path: Path):
    """Save raw wav bytes to disk."""
    if isinstance(audio_dict, dict) and audio_dict.get("bytes"):
        out_path.write_bytes(audio_dict["bytes"])
        return True
    return False


def process_parquet(local_path: str, out_dir: Path, prefix: str = "") -> list[dict]:
    """Decode parquet → wav files + metadata rows."""
    tbl = pq.read_table(local_path)
    audio_col = tbl.column("audio").to_pylist()
    inst_col = tbl.column("instrument").to_pylist()

    rows = []
    for i, (audio, inst) in enumerate(zip(audio_col, inst_col)):
        # Determine utt_id from path or generate
        if isinstance(audio, dict):
            path_field = audio.get("path") or ""
            utt_id = Path(path_field).stem if path_field else f"{prefix}_{i:06d}"
        else:
            utt_id = f"{prefix}_{i:06d}"

        out_path = out_dir / f"{utt_id}.wav"
        if not out_path.exists():
            if not save_wav_bytes(audio, out_path):
                continue

        rows.append({"utt_id": utt_id, "audio_path": str(out_path), "instrument": inst})
    return rows


def main():
    # --- Test (full) ---
    print("=== Downloading NSynth test (2 shards) ===")
    test_rows = []
    for s in range(2):
        fname = f"instrument/test-{s:05d}-of-00002.parquet"
        print(f"  {fname}...")
        local = hf_hub_download(REPO, fname, repo_type="dataset")
        test_rows.extend(process_parquet(local, TEST_DIR, prefix="test"))
    print(f"  total test wavs: {len(test_rows)}")

    # --- Train (15 shards) ---
    print(f"\n=== Downloading NSynth train ({N_TRAIN_SHARDS} shards) ===")
    train_rows = []
    for s in range(N_TRAIN_SHARDS):
        fname = f"instrument/train-{s:05d}-of-00075.parquet"
        print(f"  {fname}...")
        local = hf_hub_download(REPO, fname, repo_type="dataset")
        train_rows.extend(process_parquet(local, TRAIN_DIR, prefix=f"train{s:02d}"))
    print(f"  total train wavs: {len(train_rows)}")

    # --- Save raw rows for manifest builder ---
    import json
    (OUT_ROOT / "_test_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in test_rows) + "\n")
    (OUT_ROOT / "_train_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n")

    # Print family distribution
    from collections import Counter
    print(f"\nTest family dist: {Counter(r['instrument'] for r in test_rows)}")
    print(f"Train family dist: {Counter(r['instrument'] for r in train_rows)}")


if __name__ == "__main__":
    main()
