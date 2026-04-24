"""Smoke-check the per-modality interleave path added to data/loader.py.

Does NOT require GPU. Mirrors the wiring inside `_build_dataset`:
  - loads each modality manifest as its own streaming IterableDataset
  - per-stream shuffle with distinct seed
  - interleave_datasets with the configured probabilities + stopping_strategy
  - drains a bounded number of rows and reports the realized modality mix

If the realized mix is within tolerance of the requested probs and every
modality is observed, the HF-side wiring is correct and loader.py's new
branch will pass through to the existing processor/packer unchanged.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import Features, Sequence, Value, interleave_datasets, load_dataset

MANIFEST_DIR = Path("/mnt/tmp/listen_analysis/train_manifest")
PER_MOD = {
    "audio_asr":       MANIFEST_DIR / "asr_manifest.jsonl",
    "audio_emotion":   MANIFEST_DIR / "emotion_mcqa_manifest.jsonl",
    "audio_env_sound": MANIFEST_DIR / "env_sound_manifest.jsonl",
    "text":            MANIFEST_DIR / "text_sft_manifest.jsonl",
}
PROBS = {
    "audio_asr":       0.145,
    "audio_emotion":   0.337,
    "audio_env_sound": 0.346,
    "text":            0.172,
}
SEED = 42
BUFFER = 1000
STOPPING = "all_exhausted"
N_ROWS = 4000


# Union schema across the 4 Stage-2 manifest dialects. Each field is nullable;
# missing keys per modality materialize as None / []. Must be passed as
# `features=` to the map that normalizes the source so Arrow can unify types
# across interleave sources (otherwise list vs null casts collide).
UNION_FEATURES = Features({
    "modality":   Value("string"),
    "source":     Value("string"),
    "path":       Value("string"),
    "nubes_path": Value("string"),
    "text":       Value("string"),
    "transcript": Value("string"),
    "question":   Value("string"),
    "answer":     Value("string"),
    "label":      Value("string"),
    "rationale":  Value("string"),
    "choices":    Sequence(Value("string")),
    "labels":     Sequence(Value("string")),
})
UNION_KEYS = list(UNION_FEATURES.keys())


def _normalize(row):
    out = {}
    for k in UNION_KEYS:
        v = row.get(k)
        if k in ("choices", "labels"):
            out[k] = list(v) if v else []
        else:
            out[k] = None if v is None else str(v)
    return out


def build_source(manifest_path: Path, seed: int, tag: str):
    jsonl_files = [str(manifest_path)]
    sds = load_dataset("json", data_files=jsonl_files, split="train", streaming=True)
    sds = sds.map(_normalize, features=UNION_FEATURES)
    sds = sds.shuffle(seed=seed, buffer_size=BUFFER)
    return sds


def main() -> int:
    # Source row counts — sanity
    src_counts = {}
    for mod, path in PER_MOD.items():
        with path.open() as f:
            src_counts[mod] = sum(1 for _ in f)
    print("[src] row counts:", src_counts)

    modalities = list(PER_MOD.keys())
    probs = [PROBS[m] for m in modalities]
    print(f"[cfg] modalities={modalities} probs={probs} stopping={STOPPING} seed={SEED}")

    sub = []
    for idx, m in enumerate(modalities):
        sub_seed = SEED + idx * 1_000_003  # matches loader.py
        sub.append(build_source(PER_MOD[m], sub_seed, m))

    ds = interleave_datasets(sub, probabilities=probs, seed=SEED, stopping_strategy=STOPPING)

    seen_cols = None
    c = Counter()
    for i, row in enumerate(ds):
        if seen_cols is None:
            seen_cols = set(row.keys())
        else:
            seen_cols &= set(row.keys())
        c[row.get("modality", "<missing>")] += 1
        if i + 1 >= N_ROWS:
            break

    total = sum(c.values())
    print(f"[drain] {total} rows consumed")
    for m in modalities + [k for k in c if k not in modalities]:
        n = c.get(m, 0)
        realized = n / total if total else 0.0
        target = PROBS.get(m, float("nan"))
        delta = realized - target if isinstance(target, float) else None
        print(f"  {m:16s} n={n:5d}  realized={realized:.4f}  target={target:.4f}  delta={delta:+.4f}")

    # Tolerance: absolute 0.02 at N=4000 is ~ 1 std for p in [0.14, 0.35]
    tol = 0.02
    worst = max(abs(c.get(m, 0) / total - PROBS[m]) for m in modalities)
    missing = [m for m in modalities if c.get(m, 0) == 0]
    print(f"[result] worst |realized-target| = {worst:.4f} (tol={tol}); missing={missing}")
    print(f"[result] shared columns across modalities: {sorted(seen_cols or [])}")

    if missing:
        print("FAIL: some modality produced zero rows")
        return 1
    if worst > tol:
        print("FAIL: realized mix out of tolerance")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
