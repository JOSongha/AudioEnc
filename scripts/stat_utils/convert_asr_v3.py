"""Convert legacy ASR manifests (libri/mls/vox) to v3 spec.

Source manifests (read-only, do NOT delete):
    /mnt/ddn/omni_dataset/audio/librispeech/librispeech_<split>.jsonl
    /mnt/ddn/omni_dataset/audio/mls_english/mls_english_<split or train_part_N>.jsonl
    /mnt/ddn/omni_dataset/audio/voxpopuli/voxpopuli_<split>.jsonl

Old format:
    {"text": "...", "audio": "..."}              (most common)
    Some legacy datasets keep `audio_path` or `nubes_path` instead of `audio`.

New v3 format:
    {"modality": "audio_asr", "source": "librispeech|mls|voxpopuli",
     "audio_path": "...", "text": "..."}
    If `nubes_path` was present in the source row, it is preserved alongside
    `audio_path` (per § 포맷 스펙).

Output: /mnt/tmp/datasets/manifests/v3/<src>_<split>_<NNNN>.jsonl  (15000 rows / shard)

For MLS, train_part_1..5 is used (sum = 10,808,037 = single mls_english_train.jsonl
row count, so the parts cover all train rows). The single `mls_english_train.jsonl`
is intentionally skipped to avoid duplication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_ROWS = 15000

# (src, src_jsonl_path, out_split_label) tuples.
# `out_split_label` becomes part of the output filename:
# <src>_<out_split_label>_<NNNN>.jsonl
JOBS: list[tuple[str, str, str]] = [
    # librispeech
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_train_clean_100.jsonl", "train_clean_100"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_train_clean_360.jsonl", "train_clean_360"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_train_other_500.jsonl", "train_other_500"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_validation_clean.jsonl", "validation_clean"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_validation_other.jsonl", "validation_other"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_test_clean.jsonl", "test_clean"),
    ("librispeech", "/mnt/ddn/omni_dataset/audio/librispeech/librispeech_test_other.jsonl", "test_other"),
    # mls_english (5 train parts cover the full train split = 10,808,037 rows)
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_train_part_1.jsonl", "train_part_1"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_train_part_2.jsonl", "train_part_2"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_train_part_3.jsonl", "train_part_3"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_train_part_4.jsonl", "train_part_4"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_train_part_5.jsonl", "train_part_5"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_dev.jsonl", "dev"),
    ("mls", "/mnt/ddn/omni_dataset/audio/mls_english/mls_english_test.jsonl", "test"),
    # voxpopuli
    ("voxpopuli", "/mnt/ddn/omni_dataset/audio/voxpopuli/voxpopuli_train.jsonl", "train"),
    ("voxpopuli", "/mnt/ddn/omni_dataset/audio/voxpopuli/voxpopuli_test.jsonl", "test"),
]

SRC_NAME = {
    "librispeech": "librispeech",
    "mls": "mls",
    "voxpopuli": "voxpopuli",
}

# Map source-name → v3 `source` field.
V3_SOURCE = {
    "librispeech": "librispeech",
    "mls": "mls",
    "voxpopuli": "voxpopuli",
}


def iter_rows(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_one(src: str, in_path: Path, split_label: str) -> tuple[int, int, int]:
    """Returns (rows_written, rows_skipped, n_shards)."""
    v3_src = V3_SOURCE[src]
    out_prefix = f"{SRC_NAME[src]}_{split_label}"
    n_written = 0
    n_skipped = 0
    n_shards = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal n_shards, buf
        if not buf:
            return
        out_path = OUT_DIR / f"{out_prefix}_{n_shards:04d}.jsonl"
        with open(out_path, "w") as f:
            f.writelines(buf)
        n_shards += 1
        buf = []

    for row in iter_rows(in_path):
        text = row.get("text")
        # source row may use any of these keys for audio path
        audio_path = (
            row.get("audio_path")
            or row.get("audio")
            or row.get("path")
            or row.get("wav")
        )
        nubes_path = row.get("nubes_path")
        if not text or not str(text).strip() or (not audio_path and not nubes_path):
            n_skipped += 1
            continue
        out_row: dict = {
            "modality": "audio_asr",
            "source": v3_src,
        }
        if audio_path:
            out_row["audio_path"] = str(audio_path)
        if nubes_path:
            out_row["nubes_path"] = str(nubes_path)
        out_row["text"] = str(text).strip()

        buf.append(json.dumps(out_row, ensure_ascii=False) + "\n")
        n_written += 1
        if len(buf) >= SHARD_ROWS:
            flush()
    flush()
    return n_written, n_skipped, n_shards


def main() -> None:
    grand_total: dict[str, dict[str, int]] = {}
    for src, in_path_s, split_label in JOBS:
        in_path = Path(in_path_s)
        if not in_path.exists():
            print(f"[convert] MISSING {in_path}", flush=True)
            continue
        n_w, n_s, n_sh = convert_one(src, in_path, split_label)
        print(
            f"[convert] {src}/{split_label}: kept={n_w} skipped={n_s} "
            f"shards={n_sh}  ({in_path.name})",
            flush=True,
        )
        agg = grand_total.setdefault(src, {"rows": 0, "shards": 0})
        agg["rows"] += n_w
        agg["shards"] += n_sh

    print("[convert] === summary ===", flush=True)
    for src, agg in grand_total.items():
        print(f"  {src}: {agg['rows']} rows, {agg['shards']} shards", flush=True)


if __name__ == "__main__":
    main()
