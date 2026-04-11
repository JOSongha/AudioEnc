#!/usr/bin/env python3
"""
pack_arrow.py — per-sample precomputed Arrow → packed Arrow (offline packing)

인코더 피처가 이미 저장된 per-sample Arrow 파일을 읽어 processor_fn + packer_fn을 적용하고,
packed bin 단위 Arrow 파일로 저장한다. GPU 불필요, CPU만 사용.

출력: {precomputed_dir}/{encoder}/{dataset}/packed_{cutoff_len}/rank{N}.arrow

Arrow 스키마 (packed):
    input_ids       : list<int32>          — 길이 cutoff_len
    labels          : list<int32>          — 길이 cutoff_len
    attention_mask  : list<int32>          — 길이 cutoff_len
    audio_features  : list<list<float32>>  — bin 내 클립별 flat feature (T_enc × out_dim,)
    audio_lengths   : list<int32>          — bin 내 클립별 T_enc

사용법:
    # 단일 rank (병렬 실행 시):
    python precompute/pack_arrow.py --encoder fb_dacvae --rank 0 --num-ranks 8

    # 8 rank 병렬 (권장):
    bash precompute/run_pack.sh --encoder fb_dacvae
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import get_config
from train_pipeline_override import create_packer, make_precomputed_processor_fn

# packed Arrow 스키마
PACKED_SCHEMA = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("attention_mask", pa.list_(pa.int32())),
    pa.field("audio_features", pa.list_(pa.list_(pa.float32()))),
    pa.field("audio_lengths",  pa.list_(pa.int32())),
])

WRITE_BATCH = 500   # Arrow IPC flush 단위 (packed bin 수)


def _to_pa_table(buf: dict) -> pa.Table:
    return pa.table(
        {
            "input_ids":      pa.array(buf["input_ids"],      type=pa.list_(pa.int32())),
            "labels":         pa.array(buf["labels"],         type=pa.list_(pa.int32())),
            "attention_mask": pa.array(buf["attention_mask"], type=pa.list_(pa.int32())),
            "audio_features": pa.array(buf["audio_features"], type=pa.list_(pa.list_(pa.float32()))),
            "audio_lengths":  pa.array(buf["audio_lengths"],  type=pa.list_(pa.int32())),
        },
        schema=PACKED_SCHEMA,
    )


def pack_rank(
    ds_key: str,
    rank: int,
    base_dir: Path,
    cutoff_len: int,
    processor_fn,
    packer_fn,
    process_batch_size: int,
    packing_bucket_size: int,
):
    """하나의 (dataset, rank) 쌍에 대해 packing을 수행하고 Arrow 파일로 저장."""
    ds_dir = base_dir / ds_key

    # per-sample Arrow 파일 탐색 (sharded 우선)
    shard_files = sorted(ds_dir.glob(f"rank{rank}_s*.arrow"))
    if shard_files:
        data_files = [str(p) for p in shard_files]
    else:
        single = ds_dir / f"rank{rank}.arrow"
        if not single.exists():
            print(f"  [skip] {ds_key} rank {rank}: no Arrow files found")
            return 0
        data_files = [str(single)]

    # 출력 경로
    out_dir = ds_dir / f"packed_{cutoff_len}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank{rank}.arrow"

    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return -1

    print(f"  Processing {ds_key} rank {rank} ({len(data_files)} shard(s)) → {out_path}")

    # 스트리밍 파이프라인
    ds = load_dataset("arrow", data_files=data_files, split="train", streaming=True)
    ds = ds.map(
        processor_fn,
        batched=True,
        batch_size=process_batch_size,
        remove_columns=["utterance_id", "text", "features", "feat_len"],
    )
    ds = ds.map(
        packer_fn,
        batched=True,
        batch_size=packing_bucket_size,
    )

    buf = defaultdict(list)
    writer = None
    n_bins = 0

    for item in tqdm(ds, desc=f"{ds_key}/rank{rank}", leave=False):
        for k in PACKED_SCHEMA.names:
            buf[k].append(item[k])
        n_bins += 1

        if n_bins % WRITE_BATCH == 0:
            table = _to_pa_table(buf)
            if writer is None:
                writer = ipc.new_file(str(out_path), table.schema)
            writer.write_table(table)
            buf = defaultdict(list)

    if buf["input_ids"]:
        table = _to_pa_table(buf)
        if writer is None:
            writer = ipc.new_file(str(out_path), table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    else:
        print(f"  [warn] {ds_key} rank {rank}: no bins produced")
        return 0

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Done: {n_bins:,} bins → {out_path} ({size_mb:.0f} MB)")
    return n_bins


SHARD_MAX_BYTES = 20 * 1024 ** 3   # ~20GB per shard


def pack_rank_mixed(
    selected: list[str],
    rank: int,
    base_dir: Path,
    cutoff_len: int,
    processor_fn,
    packer_fn,
    process_batch_size: int,
    packing_bucket_size: int,
    seed: int = 42,
):
    """여러 데이터셋의 per-sample 데이터를 합쳐서 셔플 후 cross-dataset packing.

    출력: {base_dir}/mixed/packed_{cutoff_len}/rank{N}_s{S}.arrow (shard별 ~20GB)
    각 bin에 여러 데이터셋의 샘플이 섞여 들어간다.
    """
    import random

    out_dir = base_dir / "mixed" / f"packed_{cutoff_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 shard 파일이 있으면 스킵
    existing = sorted(out_dir.glob(f"rank{rank}_s*.arrow"))
    if existing:
        print(f"  [skip] {len(existing)} shard(s) already exist for rank {rank}")
        return -1

    # 1. 모든 데이터셋의 per-sample 데이터를 processor_fn으로 처리하여 수집
    all_processed = []
    for ds_key in selected:
        ds_dir = base_dir / ds_key
        shard_files = sorted(ds_dir.glob(f"rank{rank}_s*.arrow"))
        if shard_files:
            data_files = [str(p) for p in shard_files]
        else:
            single = ds_dir / f"rank{rank}.arrow"
            if not single.exists():
                print(f"  [skip] {ds_key} rank {rank}: no Arrow files found")
                continue
            data_files = [str(single)]

        print(f"  Processing {ds_key} rank {rank} ({len(data_files)} shard(s))...")
        ds = load_dataset("arrow", data_files=data_files, split="train", streaming=True)
        ds = ds.map(
            processor_fn,
            batched=True,
            batch_size=process_batch_size,
            remove_columns=["utterance_id", "text", "features", "feat_len"],
        )
        count = 0
        for item in tqdm(ds, desc=f"{ds_key}/rank{rank} (process)", leave=False):
            all_processed.append(item)
            count += 1
        print(f"    {ds_key}: {count:,} samples processed")

    if not all_processed:
        print(f"  [warn] rank {rank}: no samples collected")
        return 0

    # 2. 셔플
    print(f"  Shuffling {len(all_processed):,} samples (seed={seed})...")
    random.seed(seed + rank)
    random.shuffle(all_processed)

    # 3. 패킹 + 샤딩 (shard당 ~20GB)
    print(f"  Packing into bins (cutoff_len={cutoff_len}, bucket_size={packing_bucket_size})...")
    buf = defaultdict(list)
    writer = None
    shard_idx = 0
    shard_bytes = 0
    n_bins = 0
    total_bins = 0

    def _flush_buf():
        nonlocal writer, buf, shard_idx, shard_bytes, n_bins
        if not buf["input_ids"]:
            return
        table = _to_pa_table(buf)
        if writer is None:
            shard_path = out_dir / f"rank{rank}_s{shard_idx}.arrow"
            writer = ipc.new_file(str(shard_path), table.schema)
        writer.write_table(table)
        shard_bytes += table.nbytes
        buf = defaultdict(list)

    def _rotate_shard():
        nonlocal writer, shard_idx, shard_bytes, n_bins
        if writer:
            writer.close()
            shard_path = out_dir / f"rank{rank}_s{shard_idx}.arrow"
            size_mb = shard_path.stat().st_size / 1e6
            print(f"    Shard {shard_idx}: {n_bins:,} bins ({size_mb:.0f} MB)")
        shard_idx += 1
        shard_bytes = 0
        n_bins = 0
        writer = None

    for i in tqdm(range(0, len(all_processed), packing_bucket_size),
                  desc=f"rank{rank} (pack)", leave=False):
        batch_items = all_processed[i:i + packing_bucket_size]
        batch = defaultdict(list)
        for item in batch_items:
            for k in item:
                batch[k].append(item[k])
        packed = packer_fn(dict(batch))

        n_packed = len(packed["input_ids"])
        for j in range(n_packed):
            for k in PACKED_SCHEMA.names:
                buf[k].append(packed[k][j])
            n_bins += 1
            total_bins += 1

            if n_bins % WRITE_BATCH == 0:
                _flush_buf()
                if shard_bytes >= SHARD_MAX_BYTES:
                    _rotate_shard()

    _flush_buf()
    if writer:
        writer.close()
        shard_path = out_dir / f"rank{rank}_s{shard_idx}.arrow"
        size_mb = shard_path.stat().st_size / 1e6
        print(f"    Shard {shard_idx}: {n_bins:,} bins ({size_mb:.0f} MB)")

    del all_processed

    print(f"  Done: {total_bins:,} bins across {shard_idx + 1} shard(s)")
    return total_bins


def main():
    parser = argparse.ArgumentParser(description="Offline packing: per-sample Arrow → packed Arrow")
    parser.add_argument("--encoder",        required=True, help="인코더 이름 (e.g. fb_dacvae)")
    parser.add_argument("--rank",           type=int, required=True, help="처리할 rank (0-based)")
    parser.add_argument("--num-ranks",      type=int, default=8)
    parser.add_argument("--datasets",       default="ls100,ls360,ls500,mls,gs,vp",
                        help="쉼표 구분 데이터셋 키")
    parser.add_argument("--cutoff-len",     type=int, default=None,
                        help="packing cutoff 길이 (기본: config의 packing_cutoff_len)")
    parser.add_argument("--precomputed-dir", default="/mnt/ddn/users/jos/precomputed")
    parser.add_argument("--mixed",          action="store_true",
                        help="cross-dataset packing: 모든 데이터셋을 합쳐서 셔플 후 패킹")
    args = parser.parse_args()

    cfg = get_config(args.encoder)
    cutoff_len = args.cutoff_len or cfg["packing_cutoff_len"]

    print(f"Encoder       : {args.encoder}")
    print(f"Rank          : {args.rank} / {args.num_ranks}")
    print(f"Cutoff len    : {cutoff_len}")
    print(f"Datasets      : {args.datasets}")
    print(f"Precomputed   : {args.precomputed_dir}")
    print(f"Mixed packing : {'✓' if args.mixed else '✗'}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_model"],
        cache_dir=cfg["model_cache_dir"],
        trust_remote_code=True,
    )
    processor_fn = make_precomputed_processor_fn(cfg, tokenizer)
    packer_fn    = create_packer(
        cutoff_len=cutoff_len,
        pad_token_id=tokenizer.pad_token_id,
        neat_packing=True,
    )

    base_dir      = Path(args.precomputed_dir) / args.encoder
    selected      = [k.strip() for k in args.datasets.split(",") if k.strip()]
    total_bins    = 0

    if args.mixed:
        total_bins = pack_rank_mixed(
            selected=selected,
            rank=args.rank,
            base_dir=base_dir,
            cutoff_len=cutoff_len,
            processor_fn=processor_fn,
            packer_fn=packer_fn,
            process_batch_size=cfg.get("process_batch_size", 32),
            packing_bucket_size=cfg.get("packing_bucket_size", 200),
        )
    else:
        for ds_key in selected:
            n = pack_rank(
                ds_key=ds_key,
                rank=args.rank,
                base_dir=base_dir,
                cutoff_len=cutoff_len,
                processor_fn=processor_fn,
                packer_fn=packer_fn,
                process_batch_size=cfg.get("process_batch_size", 32),
                packing_bucket_size=cfg.get("packing_bucket_size", 200),
            )
            if n > 0:
                total_bins += n

    print(f"\nRank {args.rank} complete. Total packed bins: {total_bins:,}")


if __name__ == "__main__":
    main()
