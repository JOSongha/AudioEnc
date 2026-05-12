#!/usr/bin/env python
"""Merge per-corpus pair jsonls into a single `all_pairs.jsonl` and optionally
subsample: keep up to K transcript groups per corpus (default 60), random
seed fixed for reproducibility.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def load(p: Path):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_corpus", type=int, default=60,
                    help="max transcript groups per corpus to keep")
    ap.add_argument("--min_speakers", type=int, default=2,
                    help="drop transcript groups with < this many speakers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "all_pairs.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_rows = []
    for src_jsonl in [HERE / "iemocap_pairs.jsonl", HERE / "arctic_pairs.jsonl"]:
        if not src_jsonl.exists():
            print(f"[combine] WARN missing {src_jsonl}")
            continue
        rows = load(src_jsonl)
        by_t = defaultdict(list)
        for r in rows:
            by_t[r["transcript_id"]].append(r)
        # filter
        keys = [t for t, rs in by_t.items() if len({r["spk"] for r in rs}) >= args.min_speakers]
        rng.shuffle(keys)
        keys = keys[: args.per_corpus]
        for t in keys:
            out_rows.extend(by_t[t])
        print(f"[combine] {src_jsonl.name}: groups_kept={len(keys)} rows={sum(len(by_t[t]) for t in keys)}")

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[combine] TOTAL rows={len(out_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
