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

import numpy as np
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

    메모리 효율: processed 샘플을 temp Arrow 파일에 쓰고 memory-map으로 읽어
    셔플된 인덱스 순서로 작은 batch씩 패킹. RAM 사용량 ~수 GB (전체 올리지 않음).
    """
    import random
    import tempfile

    out_dir = base_dir / "mixed" / f"packed_{cutoff_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 shard 파일이 있으면 스킵
    existing = sorted(out_dir.glob(f"rank{rank}_s*.arrow"))
    if existing:
        print(f"  [skip] {len(existing)} shard(s) already exist for rank {rank}")
        return -1

    # Phase 1: 모든 데이터셋 스트리밍 → temp Arrow 파일에 쓰기 (RAM에 안 쌓음)
    PROCESSED_SCHEMA = pa.schema([
        pa.field("input_ids",      pa.list_(pa.int32())),
        pa.field("labels",         pa.list_(pa.int32())),
        pa.field("audio_features", pa.list_(pa.float32())),
        pa.field("audio_lengths",  pa.int32()),
    ])
    FLUSH_EVERY = 5000  # temp 파일에 flush할 단위

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        suffix=f"_rank{rank}_mixed.arrow", dir="/mnt/tmp"
    )
    import os
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)

    tmp_writer = ipc.new_file(str(tmp_path), PROCESSED_SCHEMA)
    tmp_buf = defaultdict(list)
    total_samples = 0

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
            for k in PROCESSED_SCHEMA.names:
                tmp_buf[k].append(item[k])
            count += 1
            total_samples += 1
            if count % FLUSH_EVERY == 0:
                table = pa.table({
                    "input_ids":      pa.array(tmp_buf["input_ids"],      type=pa.list_(pa.int32())),
                    "labels":         pa.array(tmp_buf["labels"],         type=pa.list_(pa.int32())),
                    "audio_features": pa.array(tmp_buf["audio_features"], type=pa.list_(pa.float32())),
                    "audio_lengths":  pa.array(tmp_buf["audio_lengths"],  type=pa.int32()),
                }, schema=PROCESSED_SCHEMA)
                tmp_writer.write_table(table)
                tmp_buf = defaultdict(list)
        # flush remaining for this dataset
        if tmp_buf["input_ids"]:
            table = pa.table({
                "input_ids":      pa.array(tmp_buf["input_ids"],      type=pa.list_(pa.int32())),
                "labels":         pa.array(tmp_buf["labels"],         type=pa.list_(pa.int32())),
                "audio_features": pa.array(tmp_buf["audio_features"], type=pa.list_(pa.float32())),
                "audio_lengths":  pa.array(tmp_buf["audio_lengths"],  type=pa.int32()),
            }, schema=PROCESSED_SCHEMA)
            tmp_writer.write_table(table)
            tmp_buf = defaultdict(list)
        print(f"    {ds_key}: {count:,} samples processed")

    tmp_writer.close()

    if total_samples == 0:
        tmp_path.unlink(missing_ok=True)
        print(f"  [warn] rank {rank}: no samples collected")
        return 0

    tmp_size_gb = tmp_path.stat().st_size / 1e9
    print(f"  Phase 1 done: {total_samples:,} samples → temp ({tmp_size_gb:.1f} GB)")

    # Phase 2: RecordBatch 오프셋 인덱스 빌드 (RAM: 인덱스 배열만)
    print(f"  Building batch index for {total_samples:,} samples...")
    mmap_file = pa.memory_map(str(tmp_path), "r")
    reader = ipc.open_file(mmap_file)

    # RecordBatch별 시작 오프셋 계산 → sample_idx → (batch_idx, row_in_batch)
    batch_offsets = []  # (cumulative_start, num_rows)
    cum = 0
    for bi in range(reader.num_record_batches):
        nr = reader.get_batch(bi).num_rows
        batch_offsets.append((cum, nr))
        cum += nr
    assert cum == total_samples, f"Mismatch: {cum} vs {total_samples}"

    indices = np.arange(total_samples, dtype=np.int64)
    rng = random.Random(seed + rank)
    # Fisher-Yates on numpy for memory efficiency
    for i in range(total_samples - 1, 0, -1):
        j = rng.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]

    print(f"  Shuffled {total_samples:,} indices (seed={seed + rank})")

    def _resolve_batch(global_idx: int):
        """global sample index → (batch_idx, row_in_batch)"""
        # binary search
        lo, hi = 0, len(batch_offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if batch_offsets[mid][0] + batch_offsets[mid][1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        return lo, global_idx - batch_offsets[lo][0]

    # Phase 3: 셔플된 인덱스 순서로 batch씩 읽어 패킹 + 샤딩
    # RecordBatch를 LRU 캐시로 관리하여 RAM 사용량 제한
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def _get_batch(batch_idx: int) -> pa.RecordBatch:
        return reader.get_batch(batch_idx)

    def _gather_rows(global_indices):
        """여러 global index에서 샘플을 모아 dict of lists로 반환."""
        cols = {k: [] for k in PROCESSED_SCHEMA.names}
        for gi in global_indices:
            bi, ri = _resolve_batch(int(gi))
            rb = _get_batch(bi)
            for k in PROCESSED_SCHEMA.names:
                cols[k].append(rb.column(k)[ri].as_py())
        return cols

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

    for i in tqdm(range(0, total_samples, packing_bucket_size),
                  desc=f"rank{rank} (pack)", leave=False):
        idx_batch = indices[i:i + packing_bucket_size]
        batch = _gather_rows(idx_batch)
        packed = packer_fn(batch)

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

    # Cleanup
    _get_batch.cache_clear()
    mmap_file.close()
    tmp_path.unlink(missing_ok=True)
    print(f"  Temp file cleaned up")

    print(f"  Done: {total_bins:,} bins across {shard_idx + 1} shard(s)")
    return total_bins


def rebalance_mixed_shards(base_dir: Path, cutoff_len: int, num_ranks: int = 8):
    """mixed packing 의 rank 간 bin 개수 불균형 보정.

    증상: 8 rank 가 독립적으로 pack_rank_mixed 를 돌면 rank 별 총 bin 수가 달라짐.
    특히 20 GB 바이트 기준 shard rotation 때문에 마지막 shard 크기가 rank 별로 편차 큼.
    이 상태로 num_data_splits=len(shards) 학습 시 split 마다 rank 마다 다른 len →
    HF Trainer 가 rank 별 다른 step 수로 돌려다가 NCCL collective 불일치 → deadlock.

    수정: 모든 rank 의 총 bin 수의 최소값으로 trim. 가장 뒤 shard 부터 bin 삭제.
    결과: 모든 rank 가 동일 bin 수, 각 shard index 도 비슷한 크기 (마지막 shard 만
    차이 있으나 학습 시 runtime truncate 로 흡수 가능).

    Args:
        base_dir: {precomputed}/{encoder}
        cutoff_len: packed/{cutoff_len} 서브디렉토리
        num_ranks: 총 rank 수

    사용: python precompute/pack_arrow.py --encoder fb_dacvae --rebalance
    """
    mixed_dir = base_dir / "mixed" / f"packed_{cutoff_len}"
    if not mixed_dir.exists():
        print(f"[rebalance] mixed dir not found: {mixed_dir}")
        return

    print(f"[rebalance] scanning {mixed_dir}")

    # rank 별 (shard_idx, path, num_rows) 수집
    # memory_map 을 쓰면 block header 를 mmap 으로 lazy 로드해서 64 shard 전체 < 1s.
    # 이전 `ipc.open_file(str(path))` 는 shard 당 ~25 s 소요 (20 GB I/O).
    rank_info = {}
    for r in range(num_ranks):
        shards = sorted(mixed_dir.glob(f"rank{r}_s*.arrow"))
        if not shards:
            print(f"  rank {r}: no shards")
            continue
        info = []
        for s in shards:
            with pa.memory_map(str(s), "r") as src:
                rdr = ipc.open_file(src)
                rows = sum(rdr.get_batch(bi).num_rows for bi in range(rdr.num_record_batches))
            info.append((s, rows))
        total = sum(rows for _, rows in info)
        rank_info[r] = (info, total)
        print(f"  rank {r}: {len(info)} shard(s), {total:,} bins", flush=True)

    if not rank_info:
        print("[rebalance] no rank data found")
        return

    min_total = min(total for _, total in rank_info.values())
    max_total = max(total for _, total in rank_info.values())
    print(f"[rebalance] min={min_total:,}  max={max_total:,}  diff={max_total - min_total:,}")

    if max_total == min_total:
        print("[rebalance] all ranks already balanced, nothing to do")
        return

    # 각 rank 를 min_total 로 trim. 가장 뒤 shard 부터 bin 삭제.
    for r, (shards, total) in rank_info.items():
        excess = total - min_total
        if excess == 0:
            continue
        print(f"[rebalance] rank {r}: trim {excess:,} bins")

        # 뒤에서부터 shard 를 순회하며 excess 만큼 rows 삭제
        for idx in range(len(shards) - 1, -1, -1):
            if excess <= 0:
                break
            shard_path, rows = shards[idx]
            if rows <= excess:
                # shard 전체 삭제
                shard_path.unlink()
                excess -= rows
                print(f"    rank {r}: removed shard {shard_path.name} ({rows} bins)")
            else:
                # shard 내부에서 rows - excess 개만 남김. batch 단위로 누적하여 new_rows 에 도달 시 중단.
                # combine_chunks() 는 audio_features (list<list<float>>) 에서 int32 offset overflow → 금지.
                # 대신 RecordBatch 를 순회하며 필요한 만큼 slice 해서 tmp 파일에 순차 write.
                new_rows = rows - excess
                with pa.memory_map(str(shard_path), "r") as src:
                    rdr = ipc.open_file(src)
                    schema = rdr.schema
                    tmp_out = shard_path.with_suffix(".tmp")
                    with ipc.new_file(str(tmp_out), schema) as wr:
                        remaining = new_rows
                        for bi in range(rdr.num_record_batches):
                            if remaining <= 0:
                                break
                            batch = rdr.get_batch(bi)
                            if batch.num_rows <= remaining:
                                wr.write_batch(batch)
                                remaining -= batch.num_rows
                            else:
                                wr.write_batch(batch.slice(0, remaining))
                                remaining = 0
                tmp_out.replace(shard_path)
                print(f"    rank {r}: trimmed {shard_path.name} {rows} → {new_rows} bins", flush=True)
                excess = 0

    # 재검증
    print("[rebalance] verifying...")
    for r in range(num_ranks):
        shards = sorted(mixed_dir.glob(f"rank{r}_s*.arrow"))
        if not shards:
            continue
        total = 0
        for s in shards:
            with pa.memory_map(str(s), "r") as src:
                rdr = ipc.open_file(src)
                for bi in range(rdr.num_record_batches):
                    total += rdr.get_batch(bi).num_rows
        print(f"  rank {r}: {total:,} bins ({len(shards)} shard(s))")
    print("[rebalance] done")


def main():
    parser = argparse.ArgumentParser(description="Offline packing: per-sample Arrow → packed Arrow")
    parser.add_argument("--encoder",        required=True, help="인코더 이름 (e.g. fb_dacvae)")
    parser.add_argument("--rank",           type=int, default=None, help="처리할 rank (0-based). --rebalance 모드에선 불필요")
    parser.add_argument("--num-ranks",      type=int, default=8)
    parser.add_argument("--datasets",       default="ls100,ls360,ls500,mls,gs,vp",
                        help="쉼표 구분 데이터셋 키")
    parser.add_argument("--cutoff-len",     type=int, default=None,
                        help="packing cutoff 길이 (기본: config의 packing_cutoff_len)")
    parser.add_argument("--precomputed-dir", default="/mnt/ddn/users/jos/precomputed")
    parser.add_argument("--mixed",          action="store_true",
                        help="cross-dataset packing: 모든 데이터셋을 합쳐서 셔플 후 패킹")
    parser.add_argument("--word-aug",       action="store_true",
                        help="word-level sub-clip 추가 (alignment 기반 features 슬라이싱). 문장 + 단어 둘 다 emit")
    parser.add_argument("--rebalance",      action="store_true",
                        help="기존 mixed shard 들의 rank 간 bin 개수 불균형 수정. packing 안 함.")
    args = parser.parse_args()

    if args.rebalance:
        cfg_tmp = get_config(args.encoder)
        cutoff_len_tmp = args.cutoff_len or cfg_tmp["packing_cutoff_len"]
        base_dir_tmp = Path(args.precomputed_dir) / args.encoder
        rebalance_mixed_shards(base_dir_tmp, cutoff_len_tmp, num_ranks=args.num_ranks)
        return

    if args.rank is None:
        parser.error("--rank 는 rebalance 모드가 아닐 때 필수")

    cfg = get_config(args.encoder)
    cutoff_len = args.cutoff_len or cfg["packing_cutoff_len"]

    print(f"Encoder       : {args.encoder}")
    print(f"Rank          : {args.rank} / {args.num_ranks}")
    print(f"Cutoff len    : {cutoff_len}")
    print(f"Datasets      : {args.datasets}")
    print(f"Precomputed   : {args.precomputed_dir}")
    print(f"Mixed packing : {'✓' if args.mixed else '✗'}")
    print(f"Word-aug      : {'✓' if args.word_aug else '✗'}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_model"],
        cache_dir=cfg["model_cache_dir"],
        trust_remote_code=True,
    )

    # word-aug: alignment lookup 빌드 (모든 선택 데이터셋의 lookup 을 MergedLookup 으로 합침)
    alignment_lookup = None
    if args.word_aug:
        from train_pipeline_override import build_alignment_lookups, MergedAlignmentLookup
        selected = [k.strip() for k in args.datasets.split(",") if k.strip()]
        per_ds = build_alignment_lookups(selected)
        alignment_lookup = MergedAlignmentLookup(list(per_ds.values()))
        print(f"AlignmentLookup: merged {sum(1 for v in per_ds.values() if v is not None)}/{len(per_ds)} datasets")

    processor_fn = make_precomputed_processor_fn(cfg, tokenizer, alignment_lookup=alignment_lookup)
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
