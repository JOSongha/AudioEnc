"""Per-epoch-random-sampled combined manifest for Stage 2 v2 training.

Motivation
----------
The original `build_combined_manifest.py` concatenates all 4 per-modality
pools into a single fixed manifest of ~118 k rows, shuffled once. The
trainer streams over this manifest; with `max_steps=50_000` × effective
batch 24 ≈ 1.2 M rows seen, the model sees each row roughly 10 times.
For text SFT (only 20 519 rows across 6 commonsense benchmarks), this
caused empirical memorization-like training-loss collapse (7.49 → 0.034
by step 12 000) — flagged in `stage2_eval_harness.md` §10.

The v2 design materializes N "pseudo-epochs", each containing:
  - emotion: ALL rows (pool is the primary task; no subsampling)
  - asr / env-sound / text: random sample of the pool, fraction-controlled

Per-epoch random sampling reduces the per-row repetition rate for the
small corpora and breaks deterministic ordering across epochs (the
trainer would otherwise see exactly the same shuffled order each pass).

Output
------
A new shard directory `stage2_combined_shards_eprandom/` containing
N_epochs × len_per_epoch / N_shards rows per file. Streaming dataloader
sees this as one big manifest; the per-epoch structure is implicit in
the row order.

Usage
-----
    python scripts/emo/build_epoch_random_manifest.py \
        --epochs 20 \
        --asr-frac 1.0 --env-frac 0.5 --text-frac 0.3 \
        --out-name stage2_combined_shards_eprandom

The defaults aim for ~75 k rows per pseudo-epoch (vs. 118 k in v1) with
emotion as 53 % of the mix and text reduced to 8 % (vs. v1's 17 %).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

BASE = Path("/mnt/tmp/listen_analysis/train_manifest")
DEFAULT_OUT_NAME = "stage2_combined_shards_eprandom"
N_SHARDS = 128
DEFAULT_SEED = 20260425

INPUTS = {
    "emotion": "emotion_mcqa_manifest.jsonl",
    "asr": "asr_manifest.jsonl",
    "env": "env_sound_manifest.jsonl",
    "text": "text_sft_manifest.jsonl",
}


def load_pool(name: str) -> list[dict]:
    p = BASE / INPUTS[name]
    if not p.exists():
        raise FileNotFoundError(f"missing manifest {p}")
    rows = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            r.setdefault("modality", {
                "emotion": "audio_emotion", "asr": "audio_asr",
                "env": "audio_env_sound", "text": "text",
            }[name])
            rows.append(r)
    return rows


def sample_without_replacement(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k >= len(rows):
        return list(rows)  # full keep
    return rng.sample(rows, k)


def build(epochs: int, fractions: dict[str, float], seed: int,
          out_name: str) -> None:
    rng_master = random.Random(seed)

    # Load pools once
    print("[v2] loading pools...")
    pools = {name: load_pool(name) for name in INPUTS}
    pool_sizes = {name: len(p) for name, p in pools.items()}

    per_epoch_quota = {
        name: (len(p) if name == "emotion"
               else round(len(p) * fractions[name]))
        for name, p in pools.items()
    }

    print(f"[v2] pool sizes: {pool_sizes}")
    print(f"[v2] per-epoch quotas (fractions={fractions}): {per_epoch_quota}")
    rows_per_epoch = sum(per_epoch_quota.values())
    print(f"[v2] rows/epoch: {rows_per_epoch:,}  total over {epochs} epochs: "
          f"{rows_per_epoch * epochs:,}")
    mix_pct = {k: v / rows_per_epoch * 100 for k, v in per_epoch_quota.items()}
    print(f"[v2] modality mix: " + " ".join(f"{k}={v:5.2f}%" for k, v in mix_pct.items()))

    # Sample per epoch and write shards interleaved (so each shard contains
    # a mix of epochs, not one epoch per shard — gives the streaming
    # dataloader a more uniform distribution per shard).
    out_dir = BASE / out_name
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("shard_*.jsonl"):
        f.unlink()
    fhs = [open(out_dir / f"shard_{i:05d}.jsonl", "w") for i in range(N_SHARDS)]

    total_written = 0
    per_modality_written = {name: 0 for name in INPUTS}

    try:
        for ep in range(epochs):
            # Per-epoch RNG so re-runs are deterministic given the same seed
            rng_ep = random.Random((seed * 1_000_003) ^ ep)
            epoch_rows: list[dict] = []
            for name, pool in pools.items():
                k = per_epoch_quota[name]
                sampled = sample_without_replacement(pool, k, rng_ep)
                epoch_rows.extend(sampled)
                per_modality_written[name] += len(sampled)
            rng_ep.shuffle(epoch_rows)
            # Round-robin into shards using a global counter to ensure
            # each shard gets ~balanced rows across epochs.
            for i, r in enumerate(epoch_rows):
                fhs[(total_written + i) % N_SHARDS].write(
                    json.dumps(r, ensure_ascii=False) + "\n"
                )
            total_written += len(epoch_rows)
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"[v2] epoch {ep+1}/{epochs} done  cum_rows={total_written:,}")
    finally:
        for f in fhs:
            f.close()

    # Report
    summary = {
        "schema": "v2-epoch-random",
        "seed": seed,
        "epochs": epochs,
        "fractions": fractions,
        "pool_sizes": pool_sizes,
        "per_epoch_quota": per_epoch_quota,
        "rows_per_epoch": rows_per_epoch,
        "total_rows": total_written,
        "per_modality_written": per_modality_written,
        "modality_mix_pct": mix_pct,
        "n_shards": N_SHARDS,
        "out_dir": str(out_dir),
    }
    with (out_dir / "build_report.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[v2] total rows written: {total_written:,}")
    print(f"[v2] per modality:")
    for k, v in per_modality_written.items():
        print(f"  {k:<8} {v:>9,}  ({v/total_written*100:5.2f}%)")
    print(f"[v2] shards: {out_dir} ({N_SHARDS} files)")
    print(f"[v2] summary: {out_dir / 'build_report.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20,
                   help="how many pseudo-epochs to materialize. With "
                        "max_steps=50k × batch=24 = 1.2 M rows seen, "
                        "20 epochs × ~75 k rows = 1.5 M provides a small buffer.")
    p.add_argument("--asr-frac", type=float, default=1.0,
                   help="ASR pool fraction sampled per epoch (default: full)")
    p.add_argument("--env-frac", type=float, default=0.5,
                   help="env-sound pool fraction per epoch (default: half)")
    p.add_argument("--text-frac", type=float, default=0.3,
                   help="text pool fraction per epoch (default: 0.3 — most "
                        "aggressive subsample to mitigate text overfit)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME)
    return p.parse_args()


def main():
    args = parse_args()
    fractions = {
        "emotion": 1.0,
        "asr": args.asr_frac,
        "env": args.env_frac,
        "text": args.text_frac,
    }
    build(args.epochs, fractions, args.seed, args.out_name)


if __name__ == "__main__":
    main()
