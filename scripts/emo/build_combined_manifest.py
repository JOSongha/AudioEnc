"""Concatenate the 4 per-category manifests into one Stage-2 training manifest.

Target mix (user-set 2026-04-24): ASR:EMO:ENV:TXT = 25:30:30:15.

Inputs (all under /mnt/tmp/listen_analysis/train_manifest/):
  asr_manifest.jsonl           {source, nubes_path, text, modality="audio_asr"}
  emotion_mcqa_manifest.jsonl  {source, path, question, choices, answer, ..., modality="audio_emotion"}
  env_sound_manifest.jsonl     {source, path, labels/captions, modality="audio_env_sound"}
  text_sft_manifest.jsonl      {source, question, choices, answer, modality="text"}

Output: /mnt/tmp/listen_analysis/train_manifest/stage2_combined_manifest.jsonl
  Rows are shuffled. Per-row `modality` drives downstream template / loader branching.
"""
import json
import random
from pathlib import Path

random.seed(20260424)
BASE = Path("/mnt/tmp/listen_analysis/train_manifest")
OUT = BASE / "stage2_combined_manifest.jsonl"

inputs = ["asr_manifest.jsonl", "emotion_mcqa_manifest.jsonl",
          "env_sound_manifest.jsonl", "text_sft_manifest.jsonl"]

all_rows = []
counts = {}
for fname in inputs:
    p = BASE / fname
    if not p.exists():
        print(f"  {fname:<35} MISSING"); continue
    with p.open() as f:
        rows = [json.loads(line) for line in f]
    # tag modality if not present
    for r in rows:
        r.setdefault("modality", "audio_unknown")
    k = rows[0].get("modality") if rows else "?"
    counts[k] = len(rows)
    print(f"  {fname:<35} n={len(rows):>7,}  modality={k}")
    all_rows.extend(rows)

random.shuffle(all_rows)
with OUT.open("w") as f:
    for r in all_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Also emit 128-shard dir so HF datasets' streaming shard(N=world_size) can split
# across ranks. A single-file manifest fails with IndexError on shard() for N>1.
SHARD_DIR = OUT.with_name("stage2_combined_shards")
SHARD_DIR.mkdir(exist_ok=True)
# Clear any prior shard files to avoid mixing old + new rows.
for old in SHARD_DIR.glob("shard_*.jsonl"):
    old.unlink()
N_SHARDS = 128
fhs = [open(SHARD_DIR / f"shard_{i:05d}.jsonl", "w") for i in range(N_SHARDS)]
for i, r in enumerate(all_rows):
    fhs[i % N_SHARDS].write(json.dumps(r, ensure_ascii=False) + "\n")
for f in fhs:
    f.close()
print(f"\ntotal rows:  {len(all_rows):,}")
print(f"shards:      {SHARD_DIR} ({N_SHARDS} files)")
print(f"modality distribution:")
for k, v in counts.items():
    print(f"  {k:<20} {v:>7,}  ({v/len(all_rows)*100:5.2f}%)")
print(f"\nmanifest: {OUT}")
