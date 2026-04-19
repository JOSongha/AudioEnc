#!/usr/bin/env python3
"""Arrow IPC 파일을 N개의 shard로 분리 (row 단위 균등 분배).

기존 rankN.arrow → rankN_s0.arrow, rankN_s1.arrow, ..., rankN_s{N-1}.arrow

Row 단위로 분배하므로 배치 수가 shard 수보다 적어도 균등하게 나뉨.
RAM에 전체 파일을 올리지 않고 2-pass streaming으로 처리.

Usage:
    python shard_arrow.py --src path/to/rank0.arrow --n-shards 4
    python shard_arrow.py --base-dir /path/to/fb_dacvae --n-shards 4 --num-ranks 8
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import pyarrow as pa


def shard_file(src: Path, n_shards: int) -> list[Path]:
    """src Arrow 파일을 n_shards개의 shard 파일로 row 균등 분배."""
    size_gb = src.stat().st_size / 1e9
    print(f"[shard] {src}  ({size_gb:.1f} GB)", flush=True)
    t0 = time.time()

    reader = pa.ipc.open_file(src)
    schema = reader.schema
    n_batches = reader.num_record_batches

    # Pass 1: total row count (배치 메타데이터만 읽으므로 빠름)
    total_rows = sum(reader.get_batch(i).num_rows for i in range(n_batches))
    rows_per_shard = math.ceil(total_rows / n_shards)
    print(f"  total_rows={total_rows}  rows_per_shard={rows_per_shard}  n_batches={n_batches}", flush=True)

    out_paths = [src.parent / f"{src.stem}_s{i}.arrow" for i in range(n_shards)]
    tmp_paths = [p.with_suffix(".arrow.tmp") for p in out_paths]

    writers = [pa.ipc.new_file(str(t), schema) for t in tmp_paths]
    row_counts = [0] * n_shards

    # Pass 2: row 단위로 shard에 균등 분배
    shard_idx = 0
    shard_rows = 0

    for i in range(n_batches):
        batch = reader.get_batch(i)
        batch_offset = 0
        rows_left = batch.num_rows

        while rows_left > 0:
            can_add = min(rows_left, rows_per_shard - shard_rows)
            sub = batch.slice(batch_offset, can_add)
            writers[shard_idx].write_batch(sub)
            row_counts[shard_idx] += can_add
            shard_rows += can_add
            batch_offset += can_add
            rows_left -= can_add

            if shard_rows >= rows_per_shard and shard_idx < n_shards - 1:
                shard_idx += 1
                shard_rows = 0

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  batch {i+1}/{n_batches}  {elapsed:.0f}s", flush=True)

    for w in writers:
        w.close()

    for tmp, out in zip(tmp_paths, out_paths):
        tmp.rename(out)

    elapsed = time.time() - t0
    print(
        f"  done {elapsed:.0f}s  "
        + "  ".join(f"s{i}={row_counts[i]}" for i in range(n_shards)),
        flush=True,
    )
    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="단일 Arrow 파일 경로")
    parser.add_argument("--base-dir", help="encoder base dir (e.g. .../fb_dacvae)")
    parser.add_argument("--datasets", default="ls100,ls360,ls500,mls,gs,vp",
                        help="쉼표 구분 dataset 키")
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--rank", type=int, default=None,
                        help="특정 rank만 처리 (--base-dir 사용 시)")
    parser.add_argument("--dataset", default=None,
                        help="특정 dataset만 처리 (--base-dir 사용 시)")
    args = parser.parse_args()

    if args.src:
        shard_file(Path(args.src), args.n_shards)
        return

    if not args.base_dir:
        print("--src 또는 --base-dir 필요", file=sys.stderr)
        sys.exit(1)

    base = Path(args.base_dir)
    datasets = [d.strip() for d in args.datasets.split(",")]
    if args.dataset:
        datasets = [args.dataset]

    ranks = list(range(args.num_ranks))
    if args.rank is not None:
        ranks = [args.rank]

    total = 0
    for ds_key in datasets:
        for rank in ranks:
            src = base / ds_key / f"rank{rank}.arrow"
            if not src.exists():
                print(f"[skip] {src} not found", flush=True)
                continue
            s0 = base / ds_key / f"rank{rank}_s0.arrow"
            if s0.exists():
                print(f"[skip] {src.name} already sharded", flush=True)
                continue
            shard_file(src, args.n_shards)
            total += 1

    print(f"\n완료: {total}개 파일 처리", flush=True)


if __name__ == "__main__":
    main()
