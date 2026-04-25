"""
Manifest duration statistics for libri_mls_vox (or any shard_*.jsonl manifest).

Strategy:
  - manifest entries have only {nubes_path, text}, no duration field.
  - Source is inferred from nubes_path keywords (mls, libri, voxpopuli).
  - Entry counts are reported per source.
  - A small random sample (--sample-n) is fetched from Nubes to estimate duration
    distribution. Requires gateway access; skip with --no-sample.

Usage:
    python scripts/inspect_manifest_duration.py
    python scripts/inspect_manifest_duration.py --max-shards 200 --sample-n 200
    python scripts/inspect_manifest_duration.py --no-sample  # counts only, offline
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"

SOURCE_KEYWORDS = {
    "mls": "mls",
    "libri": "libriTTS",
    "voxpopuli": "voxpopuli",
}


def infer_source(nubes_path: str) -> str:
    p = nubes_path.lower()
    for key, label in SOURCE_KEYWORDS.items():
        if key in p:
            return label
    return "other"


def fetch_duration_nubes(nubes_path: str, gateway: str) -> float | None:
    """Fetch audio bytes from Nubes, return duration in seconds (or None on error)."""
    try:
        import io
        import requests
        import soundfile as sf

        resp = requests.get(f"{gateway}/{nubes_path}", timeout=8.0)
        resp.raise_for_status()
        info = sf.info(io.BytesIO(resp.content))
        return info.duration
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="external/datasets/libri_mls_vox")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Limit to first N shards (default: all)")
    parser.add_argument("--sample-n", type=int, default=150,
                        help="Number of random entries to fetch from Nubes for duration estimation")
    parser.add_argument("--no-sample", action="store_true",
                        help="Skip Nubes sampling — counts only")
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--nubes-gateway", default=NUBES_GATEWAY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest)
    shards = sorted(manifest_dir.glob("shard_*.jsonl"))
    if not shards:
        print(f"No shard_*.jsonl found in {manifest_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_shards:
        shards = shards[: args.max_shards]

    print(f"Scanning {len(shards)} shard(s) in {manifest_dir} ...")

    total = 0
    source_counts: dict[str, int] = defaultdict(int)
    reservoir: list[str] = []  # reservoir sample of nubes_paths
    rng = random.Random(args.seed)

    for shard_path in shards:
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total += 1
                nubes_path = row.get("nubes_path", "")
                source = infer_source(nubes_path)
                source_counts[source] += 1

                # Reservoir sampling (k = sample_n)
                k = args.sample_n
                if nubes_path:
                    if len(reservoir) < k:
                        reservoir.append(nubes_path)
                    else:
                        j = rng.randint(0, total - 1)
                        if j < k:
                            reservoir[j] = nubes_path

    print(f"\nTotal entries: {total:,}")
    print(f"\nPer-source entry counts (inferred from nubes_path):")
    for src in sorted(source_counts.keys()):
        cnt = source_counts[src]
        pct = 100.0 * cnt / total if total else 0.0
        print(f"  {src:<15s}: {cnt:>10,}  ({pct:.1f}%)")

    # MLS utterance-level — known to be short by construction
    mls_pct = 100.0 * source_counts.get("mls", 0) / total if total else 0.0
    print(f"\nNote: MLS = utterance-level audio (typically 5-20s by dataset construction)")
    print(f"      LibriTTS-R = sentence-level (typically 2-15s)")
    print(f"      VoxPopuli = parliamentary speech (can be longer, minutes)")

    if args.no_sample or not reservoir:
        print("\nSkipping Nubes duration sampling (--no-sample or no paths found).")
        print("\nVerdict (without sampling):")
        if mls_pct > 80:
            print(f"  ✓ Dataset is {mls_pct:.0f}% MLS utterances → >30s entries expected <1%")
            print(f"    Manifest split NOT needed for Stage1.")
        else:
            print(f"  ? Cannot estimate >30s ratio without duration data.")
            print(f"    Run with --sample-n to fetch audio durations from Nubes.")
        return

    # ── Duration sampling via Nubes ───────────────────────────────────
    print(f"\nFetching duration for {len(reservoir)} sampled entries from Nubes ...")
    print(f"(gateway: {args.nubes_gateway})")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    durations: list[float] = []
    failed = 0

    def _fetch(path):
        return fetch_duration_nubes(path, args.nubes_gateway)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch, p): p for p in reservoir}
        for i, fut in enumerate(as_completed(futures)):
            d = fut.result()
            if d is not None:
                durations.append(d)
            else:
                failed += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(reservoir):
                print(f"  [{i+1}/{len(reservoir)}] fetched={len(durations)} failed={failed}", end="\r")
    print()

    if not durations:
        print("\nAll Nubes fetches failed — check gateway connectivity.")
        print(f"Try: curl {args.nubes_gateway}/{reservoir[0][:80]}")
        return

    durations.sort()
    n = len(durations)
    mean_d = sum(durations) / n
    median_d = durations[n // 2]
    over_30 = sum(1 for d in durations if d > args.threshold)
    over_25 = sum(1 for d in durations if d > 25.0)

    pct_over_30 = 100.0 * over_30 / n
    pct_over_25 = 100.0 * over_25 / n

    print(f"\nSampled duration stats ({n} entries, {failed} fetch failures):")
    print(f"  min    = {durations[0]:.2f}s")
    print(f"  mean   = {mean_d:.2f}s")
    print(f"  median = {median_d:.2f}s")
    print(f"  max    = {durations[-1]:.2f}s")
    print(f"  >30s   = {over_30}/{n} = {pct_over_30:.2f}%")
    print(f"  >25s   = {over_25}/{n} = {pct_over_25:.2f}%")

    # Extrapolate to full dataset
    estimated_over = int(total * pct_over_30 / 100.0)
    print(f"\nExtrapolated to full dataset ({total:,} entries):")
    print(f"  estimated >30s entries: ~{estimated_over:,} ({pct_over_30:.2f}%)")

    print(f"\nVerdict:")
    if pct_over_30 < 2.0:
        print(f"  ✓ >30s ratio ~{pct_over_30:.2f}% < 2% → manifest split NOT needed for Stage1")
    elif pct_over_30 < 10.0:
        print(f"  ⚠ >30s ratio ~{pct_over_30:.2f}% (2-10%) → consider split for cleaner training")
    else:
        print(f"  ✗ >30s ratio ~{pct_over_30:.2f}% > 10% → manifest split RECOMMENDED before training")


if __name__ == "__main__":
    main()
