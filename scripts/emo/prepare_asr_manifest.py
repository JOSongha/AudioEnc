"""Subsample ASR rows from the 5-corpus superset manifest.

Budget per user-set 2026-04-24 mix (ASR:EMO:ENV:TXT = 0.83:1:1:0.5):
  ASR target ≈ emotion_pool × 5/6 ≈ 34 300 rows.

Source: /mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128/
  128 shards, each row: {"nubes_path": ".../{libri,mls,vox,giga,common_voice}/...", "text": "..."}

Output: /mnt/tmp/listen_analysis/train_manifest/asr_manifest.jsonl
  Each row: {"source": "libri|mls|vox|giga|cv", "nubes_path": "...", "text": "...", "modality": "audio_asr"}

Uses Bernoulli sampling (no full count pass). nubes_path is preserved verbatim so
the loader's nubes fetch pipeline still works.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path("/mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128")
OUT = Path("/mnt/tmp/listen_analysis/train_manifest/asr_manifest.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 2026-04-25: pool expanded to ≈ emotion size (user-requested).
# Emotion pool = 39 919 → ASR target rounded to 40 000.
TARGET = 40_000
random.seed(20260425)


def classify(nubes_path: str) -> str:
    p = nubes_path.lower()
    if "/common_voice" in p or "/commonvoice" in p:
        return "cv"
    if "/gigaspeech" in p or "/giga_" in p:
        return "giga"
    if "/mls/" in p or "/mls_" in p:
        return "mls"
    if "/voxpopuli" in p or "/vox/" in p or "/vox_" in p:
        return "vox"
    if "/libri" in p or "libritts" in p:   # LibriTTS_R is the "libri" source in the superset
        return "libri"
    return "other"


def main() -> None:
    shards = sorted(SRC.glob("shard_*.jsonl"))
    print(f"scanning {len(shards)} shards in {SRC}")

    # Pass 1: count total
    total = 0
    for p in shards:
        with p.open() as f:
            for _ in f:
                total += 1
    print(f"total rows: {total:,}")

    keep_prob = min(1.0, TARGET / total * 1.15)  # over-sample 15% then trim
    print(f"keep_prob ≈ {keep_prob:.4f}")

    kept: list[dict] = []
    for p in shards:
        with p.open() as f:
            for line in f:
                if random.random() > keep_prob:
                    continue
                row = json.loads(line)
                src = classify(row["nubes_path"])
                kept.append({
                    "source": src,
                    "nubes_path": row["nubes_path"],
                    "text": row["text"],
                    "modality": "audio_asr",
                })

    random.shuffle(kept)
    kept = kept[:TARGET]

    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in kept:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print("\nper-corpus breakdown:")
    for src in sorted(counts):
        print(f"  {src:<8} {counts[src]:>6,}")
    print(f"\ntotal kept: {len(kept):,}")
    print(f"manifest:   {OUT}")


if __name__ == "__main__":
    main()
