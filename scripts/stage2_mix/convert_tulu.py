"""Convert Tulu 3 SFT mixture (filtered) → unified JSONL manifest for omni pipeline.

Filters to 5 approved general-instruction subsets (drops math/code/FLAN/WildChat/Aya/safety/etc.)
and emits {task: "text", messages: [...], source: "..."} — one shard per input parquet.
"""

import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SRC_DIR = Path("/mnt/ddn/users/sehyun/CACHE/tulu3/data")
OUT_DIR = Path("/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/text")

KEEP_SOURCES = {
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "ai2-adapt-dev/coconot_converted",
    "ai2-adapt-dev/no_robots_converted",
    "ai2-adapt-dev/oasst1_converted",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10",
}

DEFAULT_SYSTEM = "You are a helpful assistant."


def normalize_messages(messages):
    """Ensure messages start with a system turn; coerce to list[dict] form."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM}] + msgs
    return msgs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquets = sorted(glob.glob(str(SRC_DIR / "*.parquet")))
    if not parquets:
        raise FileNotFoundError(f"No parquets under {SRC_DIR}")

    total_kept = 0
    for i, pq_path in enumerate(parquets):
        table = pq.read_table(pq_path, columns=["source", "messages"]).to_pandas()
        kept = table[table["source"].isin(KEEP_SOURCES)]
        out_path = OUT_DIR / f"tulu_shard_{i:05d}.jsonl"
        n = 0
        with out_path.open("w") as f:
            for row in kept.itertuples(index=False):
                rec = {
                    "task": "text",
                    "messages": normalize_messages(row.messages),
                    "source": row.source,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        total_kept += n
        print(f"  {os.path.basename(pq_path)} → {out_path.name}: {n} rows "
              f"(filtered from {len(table)})")

    print(f"\nTotal kept: {total_kept} rows across {len(parquets)} shards")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
