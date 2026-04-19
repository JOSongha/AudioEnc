#!/usr/bin/env python3
"""
sentence_pack_arrow.py — precomputed Arrow → packed Arrow (sentence bins only).

§42 후속: half-interleave 가 projector collapse 를 개선 안 함이 확인됨. 단순히
**sentence bin 만** 생성 + cross-dataset shuffle + equal-bin shard 분할 + rank 밸런싱
을 하는 clean 버전. half_inlv_pack_arrow.py 의 사본에서 alignment / interleave /
hash_split 관련 로직 전부 제거.

포맷 (utt 당 1 bin):
  input_ids = p1 + [audio_pad]*T_proj + p2 + text_ids + [eos]
  labels    = [-100]*(|p1|+T_proj+|p2|) + text_ids + [eos]
  where p1 = "Audio:\\n", p2 = "\\nTranscript:\\n"   (§41 legacy prompt)

Pipeline:
  Phase 1: 각 dataset shard 순회 → processor_fn → tmp Arrow 누적
  Phase 2: 전체 index Fisher-Yates shuffle (seed = 42 + rank) → cross-dataset mix
  Phase 3: greedy knapsack 으로 `cutoff_len=16384` bin 채움 → 단일 intermediate Arrow
  Phase 4: bin 수 기준 N 등분 → rank{N}_s{0..S-1}.arrow
  Rebalance: rank 간 총 bin 수 ±1 로 맞춤 (NCCL timeout 방지)

사용법:
    # 단일 rank
    python precompute/sentence_pack_arrow.py --encoder fb_dacvae --rank 0 --num-ranks 8 --mixed

    # 8 rank 병렬 (wrapper)
    bash precompute/run_sentence_pack.sh --encoder fb_dacvae

출력 경로:
    {precomputed_dir}/{encoder}/mixed/packed_sentence_{cutoff_len}/rank{N}_s{S}.arrow
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
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
from train_pipeline_override import (
    IGNORE_INDEX,
    create_packer,
)

# ======================================================================
# Schemas
# ======================================================================
PROCESSED_SCHEMA = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("audio_features", pa.list_(pa.float32())),
    pa.field("audio_lengths",  pa.int32()),
])

PACKED_SCHEMA = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("attention_mask", pa.list_(pa.int32())),
    pa.field("audio_features", pa.list_(pa.list_(pa.float32()))),
    pa.field("audio_lengths",  pa.list_(pa.int32())),
])

WRITE_BATCH      = 500               # packed Arrow flush 단위
SHARD_MAX_BYTES  = 20 * 1024 ** 3    # ~20 GB per shard
FLUSH_EVERY      = 5000              # processed temp flush 단위


# ======================================================================
# processor_fn — utterance row → sentence row (§42 sentence-only)
# ======================================================================
def make_sentence_processor_fn(cfg, tokenizer):
    """processor_fn 생성. 각 utterance 를 1 sentence row 로 emit.

    출력 키 (row 당 1개):
        input_ids      : list<int>
        labels         : list<int>
        audio_features : list<float32>  — flat (T_enc × out_dim,)
        audio_lengths  : int            — T_enc

    포맷 (§41 legacy prompt):
        input_ids = p1 + [audio_pad]*T_proj + p2 + text_ids + [eos]
        labels    = [-100]*(|p1|+T_proj+|p2|) + text_ids + [eos]

    alignment / interleave / hash_split 관련 로직 전부 제거 (half_inlv 대비).
    """
    audio_pad_id = cfg.get("audio_pad_token_id", 151655)
    max_text_len = cfg.get("max_text_len", 256)
    cutoff_len   = cfg.get("packing_cutoff_len", 16384)

    proj_strides = cfg["encoder"].get("proj_strides", [2, 2])
    total_stride = 1
    for s in proj_strides:
        total_stride *= s

    eos_id = tokenizer.eos_token_id

    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)
    len_p1 = len(p1_ids)
    len_p2 = len(p2_ids)

    # GigaSpeech 의 <COMMA>, <PERIOD>, <QUESTIONMARK>, <NOISE>, <SIL>, <LAUGHTER> 등
    # literal 태그 제거. 미제거 시 Qwen tokenizer 가 ` <` + 'comma' + `>` 3 token 으로 쪼개
    # audio 와 대응 안 되는 허상 target 을 대량 생성 → loss plateau (≈3.6) 유발.
    _TAG_RE = re.compile(r"<[^>]+>")
    _WS_RE  = re.compile(r"\s+")

    def _clean_text(t: str) -> str:
        t = _TAG_RE.sub(" ", t).lower().strip()
        return _WS_RE.sub(" ", t)

    def _emit_sentence(accum, flat_feats, T_enc, text):
        if T_enc == 0:
            return
        text_ids = tokenizer.encode(_clean_text(text), add_special_tokens=False)[:max_text_len]
        if not text_ids:
            return
        T_proj = math.ceil(T_enc / total_stride)
        input_ids = p1_ids + [audio_pad_id] * T_proj + p2_ids + text_ids + [eos_id]
        labels    = (
            [-100] * len_p1
            + [-100] * T_proj
            + [-100] * len_p2
            + text_ids
            + [eos_id]
        )
        if len(input_ids) > cutoff_len:
            return
        accum["input_ids"].append(input_ids)
        accum["labels"].append(labels)
        accum["audio_features"].append(flat_feats)
        accum["audio_lengths"].append(int(T_enc))

    def process_samples(examples):
        accum = {
            "input_ids":      [],
            "labels":         [],
            "audio_features": [],
            "audio_lengths":  [],
        }
        # utterance_id 는 사용 안 하지만 compat 를 위해 허용.
        n = len(examples["features"])
        for i in range(n):
            flat_feats = examples["features"][i]
            feat_len   = examples["feat_len"][i]
            text       = examples["text"][i]
            if feat_len == 0 or not flat_feats:
                continue
            _emit_sentence(accum, flat_feats, int(feat_len), text)

        return accum

    return process_samples


# ======================================================================
# Worker-process state for `--num-workers > 1` phase 1 parallelization.
#
# 각 worker 가 자기 tokenizer + alignment_lookup + processor_fn + mmap reader 를
# module 전역으로 캐시. main 은 (fpath, batch_idx) 만 dispatch → worker 가 mmap
# 에서 RecordBatch 를 직접 읽고 processor_fn 까지 돌림.
# 이렇게 해야 pyarrow→Python 변환까지 병렬 가속됨.
# ======================================================================
_WORKER_PROCESSOR = None      # callable(chunk_dict) -> processed_dict
_WORKER_READERS: dict[str, object] = {}   # fpath → ipc.RecordBatchFileReader
_WORKER_MMAPS:   dict[str, object] = {}   # fpath → pa.MemoryMappedFile (keep alive)
_WORKER_COLS = ("utterance_id", "features", "feat_len", "text")


def _phase1_worker_init(encoder_name: str, datasets_selected: list[str]):
    """ProcessPool worker 초기화. 각 자식 프로세스에서 한 번 실행."""
    global _WORKER_PROCESSOR
    cfg = get_config(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_model"],
        cache_dir=cfg["model_cache_dir"],
        trust_remote_code=True,
    )
    _WORKER_PROCESSOR = make_sentence_processor_fn(cfg, tokenizer)


def _worker_get_reader(fpath: str):
    rdr = _WORKER_READERS.get(fpath)
    if rdr is None:
        mmap = pa.memory_map(fpath, "r")
        _WORKER_MMAPS[fpath] = mmap        # mmap 객체는 살려둬야 unmap 안 됨
        rdr = ipc.open_file(mmap)
        _WORKER_READERS[fpath] = rdr
    return rdr


def _phase1_worker_process_batch(args):
    """(fpath, batch_idx) → processor_fn 출력 dict.

    pyarrow→Python 변환 과 processor_fn 모두 worker 쪽에서 실행 → 진짜 병렬.
    """
    fpath, batch_idx = args
    rdr = _worker_get_reader(fpath)
    batch = rdr.get_batch(batch_idx)
    chunk = {k: batch.column(k).to_pylist() for k in _WORKER_COLS}
    return _WORKER_PROCESSOR(chunk)


def _enumerate_arrow_batches(data_files: list[str]) -> list[tuple[str, int]]:
    """각 데이터 파일의 RecordBatch index 를 평탄화한 리스트로 반환.

    main 이 이 리스트를 pool 에 dispatch 하면, 각 task 는 worker 쪽에서
    fpath/batch_idx 를 mmap 으로 랜덤 접근해서 처리.
    """
    items: list[tuple[str, int]] = []
    for fpath in data_files:
        with pa.memory_map(fpath, "r") as src:
            rdr = ipc.open_file(src)
            n = rdr.num_record_batches
        items.extend((fpath, bi) for bi in range(n))
    return items


def _run_phase1_parallel(
    selected: list[str],
    rank: int,
    base_dir: Path,
    encoder_name: str,
    num_workers: int,
    chunk_size: int,
    tmp_writer,
    tmp_buf: dict,
    flush_every_rows: int,
) -> int:
    """batch-index dispatch 방식으로 phase 1 병렬 실행.

    `chunk_size` 는 참고용(실 chunk 크기 = 원본 RecordBatch row 수).
    """
    del chunk_size  # not used in batch-index mode

    pool = mp.get_context("spawn").Pool(
        processes=num_workers,
        initializer=_phase1_worker_init,
        initargs=(encoder_name, selected),
    )

    total_samples = 0
    try:
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

            batch_args = _enumerate_arrow_batches(data_files)
            print(
                f"  Processing {ds_key} rank {rank} ({len(data_files)} shard(s), "
                f"{len(batch_args)} record batches) [parallel × {num_workers}]..."
            )
            count = 0
            for result in tqdm(
                pool.imap_unordered(
                    _phase1_worker_process_batch, batch_args, chunksize=1
                ),
                total=len(batch_args),
                desc=f"{ds_key}/rank{rank} (process)",
                leave=False,
            ):
                m = len(result["input_ids"])
                if m == 0:
                    continue
                for k in PROCESSED_SCHEMA.names:
                    tmp_buf[k].extend(result[k])
                count += m
                total_samples += m
                if len(tmp_buf["input_ids"]) >= flush_every_rows:
                    tmp_writer.write_table(_to_processed_table(tmp_buf))
                    for k in tmp_buf:
                        tmp_buf[k].clear()
            if tmp_buf["input_ids"]:
                tmp_writer.write_table(_to_processed_table(tmp_buf))
                for k in tmp_buf:
                    tmp_buf[k].clear()
            print(f"    {ds_key}: {count:,} samples processed")
    finally:
        pool.close()
        pool.join()

    return total_samples


# ======================================================================
# Arrow packed writer helpers (archived pack_arrow.py 와 동일)
# ======================================================================
def _to_packed_table(buf: dict) -> pa.Table:
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


def _to_processed_table(buf: dict) -> pa.Table:
    return pa.table(
        {
            "input_ids":      pa.array(buf["input_ids"],      type=pa.list_(pa.int32())),
            "labels":         pa.array(buf["labels"],         type=pa.list_(pa.int32())),
            "audio_features": pa.array(buf["audio_features"], type=pa.list_(pa.float32())),
            "audio_lengths":  pa.array(buf["audio_lengths"],  type=pa.int32()),
        },
        schema=PROCESSED_SCHEMA,
    )


# ======================================================================
# Per-dataset pack (non-mixed 모드)
# ======================================================================
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
    ds_dir = base_dir / ds_key
    shard_files = sorted(ds_dir.glob(f"rank{rank}_s*.arrow"))
    if shard_files:
        data_files = [str(p) for p in shard_files]
    else:
        single = ds_dir / f"rank{rank}.arrow"
        if not single.exists():
            print(f"  [skip] {ds_key} rank {rank}: no Arrow files found")
            return 0
        data_files = [str(single)]

    out_dir = ds_dir / f"packed_sentence_{cutoff_len}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank{rank}.arrow"

    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return -1

    print(f"  Processing {ds_key} rank {rank} ({len(data_files)} shard(s)) → {out_path}")

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
            table = _to_packed_table(buf)
            if writer is None:
                writer = ipc.new_file(str(out_path), table.schema)
            writer.write_table(table)
            buf = defaultdict(list)

    if buf["input_ids"]:
        table = _to_packed_table(buf)
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


# ======================================================================
# Mixed-dataset pack (cross-dataset shuffle → shard)
# ======================================================================
def _resplit_into_equal_shards(
    src_path: Path, out_dir: Path, rank: int, num_shards: int
) -> int:
    """단일 packed arrow 파일을 num_shards 개 동일 bin 수 shard 로 분할.

    총 bin 수를 num_shards 로 나눠 각 shard 목표 크기 결정 (나머지는 앞쪽 shard 들에
    1 씩 분배). RecordBatch 단위로 읽으면서 slice 해서 순차 write. 원본 파일은 삭제.
    반환: 실제 생성된 shard 개수.
    """
    with pa.memory_map(str(src_path), "r") as src:
        rdr = ipc.open_file(src)
        schema = rdr.schema
        total_bins = sum(
            rdr.get_batch(bi).num_rows for bi in range(rdr.num_record_batches)
        )

    if total_bins == 0:
        src_path.unlink(missing_ok=True)
        return 0

    num_shards = max(1, min(num_shards, total_bins))
    base = total_bins // num_shards
    rem = total_bins - base * num_shards
    sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
    assert sum(sizes) == total_bins

    print(f"    resplit: {total_bins:,} bins → {num_shards} shard(s) "
          f"({sizes[0]:,}{' each' if rem == 0 else f' or {sizes[0]-1:,}'})")

    with pa.memory_map(str(src_path), "r") as src:
        rdr = ipc.open_file(src)
        shard_idx = 0
        written_in_shard = 0
        target = sizes[0]
        shard_path = out_dir / f"rank{rank}_s{shard_idx}.arrow"
        writer = ipc.new_file(str(shard_path), schema)

        for bi in range(rdr.num_record_batches):
            batch = rdr.get_batch(bi)
            n = batch.num_rows
            offset = 0
            while offset < n:
                remaining = target - written_in_shard
                if remaining <= 0:
                    writer.close()
                    size_mb = shard_path.stat().st_size / 1e6
                    print(f"      shard {shard_idx}: {written_in_shard:,} bins "
                          f"({size_mb:.0f} MB)")
                    shard_idx += 1
                    if shard_idx >= num_shards:
                        break
                    target = sizes[shard_idx]
                    written_in_shard = 0
                    shard_path = out_dir / f"rank{rank}_s{shard_idx}.arrow"
                    writer = ipc.new_file(str(shard_path), schema)
                    continue
                take = min(n - offset, remaining)
                writer.write_batch(batch.slice(offset, take))
                written_in_shard += take
                offset += take
            if shard_idx >= num_shards:
                break

        writer.close()
        size_mb = shard_path.stat().st_size / 1e6
        print(f"      shard {shard_idx}: {written_in_shard:,} bins ({size_mb:.0f} MB)")

    src_path.unlink(missing_ok=True)
    return shard_idx + 1


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
    num_workers: int = 1,
    encoder_name: str | None = None,
    shards_per_rank: int = 8,
    pack_subdir: str | None = None,
):
    out_dir = base_dir / "mixed" / (pack_subdir or f"packed_sentence_{cutoff_len}")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"rank{rank}_s*.arrow"))
    if existing:
        print(f"  [skip] {len(existing)} shard(s) already exist for rank {rank}")
        return -1

    # Phase 1: 모든 데이터셋 processor_fn 통과 → temp Arrow 파일에 쓰기
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        suffix=f"_rank{rank}_sentence.arrow", dir="/mnt/tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)

    tmp_writer = ipc.new_file(str(tmp_path), PROCESSED_SCHEMA)
    tmp_buf = defaultdict(list)
    total_samples = 0

    if num_workers and num_workers > 1:
        # Parallel phase 1: ProcessPoolExecutor, pyarrow 직접 iterate.
        assert encoder_name is not None, "encoder_name required for parallel phase 1"
        total_samples = _run_phase1_parallel(
            selected=selected,
            rank=rank,
            base_dir=base_dir,
            encoder_name=encoder_name,
            num_workers=num_workers,
            chunk_size=process_batch_size,
            tmp_writer=tmp_writer,
            tmp_buf=tmp_buf,
            flush_every_rows=FLUSH_EVERY,
        )
    else:
        # Sequential phase 1 (기존 경로 유지)
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
                    tmp_writer.write_table(_to_processed_table(tmp_buf))
                    tmp_buf = defaultdict(list)
            if tmp_buf["input_ids"]:
                tmp_writer.write_table(_to_processed_table(tmp_buf))
                tmp_buf = defaultdict(list)
            print(f"    {ds_key}: {count:,} samples processed")

    tmp_writer.close()

    if total_samples == 0:
        tmp_path.unlink(missing_ok=True)
        print(f"  [warn] rank {rank}: no samples collected")
        return 0

    tmp_size_gb = tmp_path.stat().st_size / 1e9
    print(f"  Phase 1 done: {total_samples:,} samples → temp ({tmp_size_gb:.1f} GB)")

    # Phase 2: Fisher-Yates shuffle
    print(f"  Building batch index for {total_samples:,} samples...")
    mmap_file = pa.memory_map(str(tmp_path), "r")
    reader = ipc.open_file(mmap_file)

    batch_offsets = []   # (cumulative_start, num_rows)
    cum = 0
    for bi in range(reader.num_record_batches):
        nr = reader.get_batch(bi).num_rows
        batch_offsets.append((cum, nr))
        cum += nr
    assert cum == total_samples, f"Mismatch: {cum} vs {total_samples}"

    indices = np.arange(total_samples, dtype=np.int64)
    rng = random.Random(seed + rank)
    for i in range(total_samples - 1, 0, -1):
        j = rng.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]

    print(f"  Shuffled {total_samples:,} indices (seed={seed + rank})")

    def _resolve_batch(global_idx: int):
        lo, hi = 0, len(batch_offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if batch_offsets[mid][0] + batch_offsets[mid][1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        return lo, global_idx - batch_offsets[lo][0]

    @lru_cache(maxsize=4)
    def _get_batch(batch_idx: int) -> pa.RecordBatch:
        return reader.get_batch(batch_idx)

    def _gather_rows(global_indices):
        cols = {k: [] for k in PROCESSED_SCHEMA.names}
        for gi in global_indices:
            bi, ri = _resolve_batch(int(gi))
            rb = _get_batch(bi)
            for k in PROCESSED_SCHEMA.names:
                cols[k].append(rb.column(k)[ri].as_py())
        return cols

    # Phase 3: pack into ONE intermediate shard (rank{N}_pack.arrow)
    # Phase 4: resplit into shards_per_rank 개의 동일 bin 수 샤드로 split.
    #
    # (기존 byte-size 기반 rotation 은 shard 간 bin 개수 불균등 → 학습 측 num_data_splits
    #  iteration 에서 split 별 step 수 편차 발생. bin 기준 equal split 로 해결.)
    pack_intermediate = out_dir / f"rank{rank}_pack.arrow"
    print(f"  Packing into single intermediate (cutoff_len={cutoff_len}, bucket_size={packing_bucket_size})...")
    buf = defaultdict(list)
    writer = None
    n_bins = 0
    total_bins = 0

    def _flush_buf():
        nonlocal writer, buf
        if not buf["input_ids"]:
            return
        table = _to_packed_table(buf)
        if writer is None:
            writer = ipc.new_file(str(pack_intermediate), table.schema)
        writer.write_table(table)
        buf = defaultdict(list)

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

    _flush_buf()
    if writer:
        writer.close()

    _get_batch.cache_clear()
    mmap_file.close()
    tmp_path.unlink(missing_ok=True)
    print("  Temp file cleaned up")

    # Phase 4: resplit into equal shards (bin count basis)
    if total_bins == 0:
        pack_intermediate.unlink(missing_ok=True)
        print(f"  [warn] rank {rank}: 0 bins packed")
        return 0

    n_actual_shards = _resplit_into_equal_shards(
        pack_intermediate, out_dir, rank=rank, num_shards=shards_per_rank
    )
    print(f"  Done: {total_bins:,} bins across {n_actual_shards} shard(s)")
    return total_bins


# ======================================================================
# Rank rebalance (§26 동일 로직, 출력 디렉토리만 packed_sentence_*)
# ======================================================================
def rebalance_mixed_shards(base_dir: Path, cutoff_len: int, num_ranks: int = 8,
                           pack_subdir: str | None = None):
    """
    Redistribute 방식: 초과 rank 의 tail bin 을 부족 rank 로 이동 (데이터 손실 없음).

    target = grand_total // n_ranks 기본, 나머지 r 명은 +1 bin.
    초과 rank: tail shard 에서 excess bin 제거 (shard rewrite).
    부족 rank: 새 shard (rank{R}_s{next_idx}.arrow) 생성 후 overflow 받음.
    """
    mixed_dir = base_dir / "mixed" / (pack_subdir or f"packed_sentence_{cutoff_len}")
    if not mixed_dir.exists():
        print(f"[rebalance] mixed dir not found: {mixed_dir}")
        return

    print(f"[rebalance] scanning {mixed_dir}")

    rank_info = {}
    schema = None
    for r in range(num_ranks):
        shards = sorted(mixed_dir.glob(f"rank{r}_s*.arrow"))
        if not shards:
            print(f"  rank {r}: no shards")
            continue
        info = []
        for s in shards:
            with pa.memory_map(str(s), "r") as src:
                rdr = ipc.open_file(src)
                if schema is None:
                    schema = rdr.schema
                rows = sum(rdr.get_batch(bi).num_rows for bi in range(rdr.num_record_batches))
            info.append((s, rows))
        total = sum(rows for _, rows in info)
        rank_info[r] = (info, total)
        print(f"  rank {r}: {len(info)} shard(s), {total:,} bins", flush=True)

    if not rank_info:
        print("[rebalance] no rank data found")
        return

    grand_total = sum(total for _, total in rank_info.values())
    n_ranks_real = len(rank_info)
    base_target = grand_total // n_ranks_real
    remainder   = grand_total - base_target * n_ranks_real  # 0 ~ n_ranks-1

    sorted_ranks = sorted(rank_info.keys())
    # 모든 rank 정확히 base_target bin (DDP 동기화 안정).
    # leftover `remainder` bin 은 _remainder.arrow 로 빠짐 → 추후 단일 GPU 1 회 추가 run 용.
    targets = {r: base_target for r in sorted_ranks}

    diffs = {r: rank_info[r][1] - targets[r] for r in sorted_ranks}
    print(f"[rebalance] total={grand_total:,}  target={base_target:,} (×{n_ranks_real})"
          + (f"  leftover={remainder} → _remainder.arrow" if remainder else ""))

    if all(d == 0 for d in diffs.values()) and remainder == 0:
        print("[rebalance] already balanced, nothing to do")
        return

    donors    = [(r, diffs[r])  for r in sorted_ranks if diffs[r] > 0]
    acceptors = [(r, -diffs[r]) for r in sorted_ranks if diffs[r] < 0]

    print(f"[rebalance] donors: {[(r, e) for r, e in donors]}")
    print(f"[rebalance] acceptors: {[(r, d) for r, d in acceptors]}")

    # Phase 1: donor tail bins → single overflow arrow file
    overflow_path = mixed_dir / "_overflow.arrow"
    if overflow_path.exists():
        overflow_path.unlink()

    with ipc.new_file(str(overflow_path), schema) as ovw:
        for r, excess in donors:
            shards, _ = rank_info[r]
            remaining = excess
            for idx in range(len(shards) - 1, -1, -1):
                if remaining <= 0:
                    break
                shard_path, rows = shards[idx]
                if rows <= remaining:
                    with pa.memory_map(str(shard_path), "r") as src:
                        rdr = ipc.open_file(src)
                        for bi in range(rdr.num_record_batches):
                            ovw.write_batch(rdr.get_batch(bi))
                    shard_path.unlink()
                    remaining -= rows
                    print(f"    rank {r}: moved whole shard {shard_path.name} ({rows} bins) → overflow", flush=True)
                else:
                    keep_rows = rows - remaining
                    tmp_path = shard_path.with_suffix(".tmp")
                    with pa.memory_map(str(shard_path), "r") as src:
                        rdr = ipc.open_file(src)
                        with ipc.new_file(str(tmp_path), schema) as kw:
                            seen = 0
                            for bi in range(rdr.num_record_batches):
                                batch = rdr.get_batch(bi)
                                if seen + batch.num_rows <= keep_rows:
                                    kw.write_batch(batch)
                                    seen += batch.num_rows
                                elif seen >= keep_rows:
                                    ovw.write_batch(batch)
                                else:
                                    split = keep_rows - seen
                                    kw.write_batch(batch.slice(0, split))
                                    ovw.write_batch(batch.slice(split))
                                    seen = keep_rows
                    tmp_path.replace(shard_path)
                    print(f"    rank {r}: split {shard_path.name}: kept {keep_rows}, moved {remaining} → overflow", flush=True)
                    remaining = 0

    # Phase 2: overflow → new shard at each acceptor
    with pa.memory_map(str(overflow_path), "r") as ovs:
        ov_rdr = ipc.open_file(ovs)
        ov_batches = [ov_rdr.get_batch(bi) for bi in range(ov_rdr.num_record_batches)]
        ov_bi, ov_off = 0, 0

        for r, deficit in acceptors:
            shards, _ = rank_info[r]
            max_idx = -1
            for s, _ in shards:
                try:
                    max_idx = max(max_idx, int(s.stem.split("_s")[1]))
                except (IndexError, ValueError):
                    pass
            new_shard = mixed_dir / f"rank{r}_s{max_idx + 1}.arrow"

            needed = deficit
            with ipc.new_file(str(new_shard), schema) as acw:
                while needed > 0 and ov_bi < len(ov_batches):
                    batch = ov_batches[ov_bi]
                    avail = batch.num_rows - ov_off
                    if avail <= needed:
                        acw.write_batch(batch.slice(ov_off))
                        needed -= avail
                        ov_bi += 1
                        ov_off = 0
                    else:
                        acw.write_batch(batch.slice(ov_off, needed))
                        ov_off += needed
                        needed = 0
            print(f"    rank {r}: wrote {new_shard.name} +{deficit - needed} bins", flush=True)

        # Acceptor 다 채운 뒤 남은 overflow = 진짜 leftover.
        # 별도 _remainder.arrow 로 빼서 drop 없이 보존 (단일 GPU 추가 run 용).
        leftover = sum(b.num_rows for b in ov_batches[ov_bi:]) - ov_off
        if leftover > 0:
            remainder_path = mixed_dir / "_remainder.arrow"
            if remainder_path.exists():
                remainder_path.unlink()
            with ipc.new_file(str(remainder_path), schema) as rmw:
                # 첫 번째 남은 batch 는 ov_off 부터
                if ov_bi < len(ov_batches) and ov_off > 0:
                    rmw.write_batch(ov_batches[ov_bi].slice(ov_off))
                    ov_bi += 1
                for bi in range(ov_bi, len(ov_batches)):
                    rmw.write_batch(ov_batches[bi])
            print(f"    _remainder.arrow: {leftover} bins (단일 GPU 추가 run 전용, drop 없음)", flush=True)
        elif leftover < 0:
            raise RuntimeError(f"[rebalance] negative leftover: {leftover}")

    overflow_path.unlink()

    print("[rebalance] verifying...")
    verified_total = 0
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
        verified_total += total
        print(f"  rank {r}: {total:,} bins ({len(shards)} shard(s))")
    rem_path = mixed_dir / "_remainder.arrow"
    if rem_path.exists():
        with pa.memory_map(str(rem_path), "r") as src:
            rdr = ipc.open_file(src)
            rem = sum(rdr.get_batch(bi).num_rows for bi in range(rdr.num_record_batches))
        verified_total += rem
        print(f"  _remainder: {rem} bins")
    print(f"[rebalance] grand_total check: {verified_total:,} vs input {grand_total:,} "
          + ("OK" if verified_total == grand_total else "MISMATCH!"))
    print("[rebalance] done")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Half-interleave packing: utt 당 hash split 로 sentence/interleave 1:1"
    )
    parser.add_argument("--encoder",         required=True, help="encoder name (e.g. fb_dacvae)")
    parser.add_argument("--rank",            type=int, default=None,
                        help="처리할 rank (0-based). --rebalance 모드에선 불필요")
    parser.add_argument("--num-ranks",       type=int, default=8)
    parser.add_argument("--datasets",        default="ls100,ls360,ls500,mls,gs,vp",
                        help="쉼표 구분 데이터셋 키")
    parser.add_argument("--cutoff-len",      type=int, default=None,
                        help="packing cutoff 길이 (기본: config 의 packing_cutoff_len)")
    parser.add_argument("--precomputed-dir", default="/mnt/ddn/users/jos/precomputed")
    parser.add_argument("--mixed",           action="store_true",
                        help="cross-dataset packing: 모든 데이터셋을 합쳐서 셔플 후 패킹")
    parser.add_argument("--num-workers",     type=int, default=1,
                        help="phase 1 processor 병렬화 워커 수 (rank 내부 멀티프로세스). "
                             "기본 1 (sequential). 4~8 권장. rank 수 × num_workers 가 총 코어.")
    parser.add_argument("--shards-per-rank", type=int, default=8,
                        help="rank 당 출력 shard 개수. 2-pass 로 bin 수를 동일하게 분할 "
                             "(마지막 shard 도 ±1 이내). 기본 8 (num_data_splits 와 매칭).")
    parser.add_argument("--rebalance",       action="store_true",
                        help="기존 mixed shard 들의 rank 간 bin 개수 불균형 수정. packing 안 함.")
    args = parser.parse_args()

    if args.rebalance:
        cfg_tmp = get_config(args.encoder)
        cutoff_len_tmp = args.cutoff_len or cfg_tmp["packing_cutoff_len"]
        base_dir_tmp = Path(args.precomputed_dir) / args.encoder
        rebalance_mixed_shards(
            base_dir_tmp, cutoff_len_tmp, num_ranks=args.num_ranks,
        )
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
    print(f"Num workers   : {args.num_workers} (phase 1 parallelism)")
    print(f"Shards/rank   : {args.shards_per_rank} (equal bin count per shard)")
    print(f"Mode          : sentence-only (§42, no alignment/interleave)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_model"],
        cache_dir=cfg["model_cache_dir"],
        trust_remote_code=True,
    )

    selected = [k.strip() for k in args.datasets.split(",") if k.strip()]

    processor_fn = make_sentence_processor_fn(cfg, tokenizer)
    packer_fn    = create_packer(
        cutoff_len=cutoff_len,
        pad_token_id=tokenizer.pad_token_id,
        neat_packing=True,
    )

    base_dir   = Path(args.precomputed_dir) / args.encoder
    total_bins = 0

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
            num_workers=args.num_workers,
            encoder_name=args.encoder,
            shards_per_rank=args.shards_per_rank,
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
