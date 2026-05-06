"""Extract LAION-Audio-630K Freesound (no_overlap) webdataset shards and build training manifest.

Source: HF dataset Meranti/CLAP_freesound, subdir freesound_no_overlap/{train_1,train_2,test}/*.tar
Each .tar: ~512 sample pairs of (mnt/freesound/split/train/{id}.flac, mnt/freesound/split/train/{id}.json)
JSON schema: {"text": [cap1, cap2?], "tag": [...], "original_data": {...}}

Output:
- Extracted .flac files at: <DATA>/extracted/freesound_no_overlap/{split}/{id}.flac
- Manifest jsonl row format:
    {"path": ".../{id}.flac",
     "source": "laion_freesound",
     "modality": "audio_env_sound",
     "captions": [cap1, cap2]}     # filtered: drop captions that are pure-numeric / very short

Idempotent: re-running skips already-extracted ids; manifest is rebuilt from scratch each run.

Usage:
    python scripts/env_sound/prepare_laion_freesound_manifest.py \
        [--splits train_1 train_2] [--keep-tars] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA_ROOT = Path("/mnt/tmp/datasets/laion_audio")
SHARDS_ROOT = DATA_ROOT / "freesound_no_overlap"
EXTRACT_ROOT = DATA_ROOT / "extracted" / "freesound_no_overlap"
MANIFEST_PATH = Path("/mnt/tmp/listen_analysis/train_manifest/laion_freesound_manifest.jsonl")

# Captions to drop: pure-numeric titles ("4.", "001"), very short, or filename-like
_NUM_RE = re.compile(r"^[\d\.\s\-_]+$")


def _is_useful_caption(c: str) -> bool:
    if not c or not isinstance(c, str):
        return False
    c = c.strip()
    if len(c) < 5:
        return False
    if _NUM_RE.match(c):
        return False
    return True


def extract_one_tar(tar_path: Path, out_dir: Path) -> list[dict]:
    """Extract a single tar; return list of manifest rows."""
    rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()

        # Build {id: {flac_member, json_member}} pairs from member names
        # Member names look like: mnt/freesound/split/train/{id}.{flac,json}
        pairs: dict[str, dict] = {}
        for m in members:
            if not m.isfile():
                continue
            name = m.name
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            if ext == ".flac":
                pairs.setdefault(stem, {})["flac"] = m
            elif ext == ".json":
                pairs.setdefault(stem, {})["json"] = m

        for fid, p in pairs.items():
            if "flac" not in p or "json" not in p:
                continue
            flac_out = out_dir / f"{fid}.flac"

            # Parse JSON for captions
            try:
                jf = tf.extractfile(p["json"])
                if jf is None:
                    continue
                meta = json.loads(jf.read().decode("utf-8"))
            except Exception:
                continue
            caps_raw = meta.get("text") or []
            if isinstance(caps_raw, str):
                caps_raw = [caps_raw]
            caps = [c.strip() for c in caps_raw if _is_useful_caption(c)]
            if not caps:
                continue

            # Extract flac if not already present
            if not flac_out.exists():
                try:
                    fobj = tf.extractfile(p["flac"])
                    if fobj is None:
                        continue
                    with open(flac_out, "wb") as out_f:
                        # Stream copy in 64K chunks
                        while True:
                            chunk = fobj.read(65536)
                            if not chunk:
                                break
                            out_f.write(chunk)
                except Exception:
                    if flac_out.exists():
                        flac_out.unlink(missing_ok=True)
                    continue

            rows.append({
                "path": str(flac_out),
                "source": "laion_freesound",
                "modality": "audio_env_sound",
                "captions": caps,
            })
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train_1", "train_2"],
                   choices=["train_1", "train_2", "test"],
                   help="Which subdirs of freesound_no_overlap to process.")
    p.add_argument("--keep-tars", action="store_true",
                   help="Keep .tar files after extraction. Default: keep.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel tar extractions. Tune for I/O bandwidth.")
    return p.parse_args()


def main():
    args = parse_args()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    out_split_dir = EXTRACT_ROOT / "train"  # all train_1+train_2 → flat train/

    tar_files: list[Path] = []
    for sp in args.splits:
        d = SHARDS_ROOT / sp
        if not d.exists():
            print(f"[laion] WARN missing dir: {d} (skip)")
            continue
        tar_files.extend(sorted(d.glob("*.tar")))
    print(f"[laion] {len(tar_files)} tar shards to process across splits={args.splits}")

    rows_total: list[dict] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_one_tar, tp, out_split_dir): tp for tp in tar_files}
        for fut in as_completed(futs):
            tp = futs[fut]
            try:
                rows = fut.result()
                rows_total.extend(rows)
            except Exception as e:
                print(f"[laion] ERROR {tp.name}: {e}")
                continue
            completed += 1
            if completed % 25 == 0 or completed == len(tar_files):
                print(f"[laion] {completed}/{len(tar_files)} shards  rows so far: {len(rows_total):,}")

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for r in rows_total:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[laion] wrote {len(rows_total):,} rows -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
