#!/usr/bin/env python3
"""
half_inlv_pack_arrow.py — precomputed Arrow → packed Arrow with 50/50 sentence / interleave split.

각 utterance 에 대해 `hash(utt_id) & 1` 결정 (md5 기반 → 결정론적):
  0: sentence bin 만 emit  — 기존 포맷 `[audio_pad]*T_proj + text_ids + EOS`
  1: interleave bin 만 emit — 단어별 교차 포맷:
       `[audio_pad]*T_proj_w0 + <|audio_correspond|> + text_w0 +
        [audio_pad]*T_proj_w1 + <|audio_correspond|> + text_w1 + ... + EOS`

alignment 가 없거나 word 수 < 2 면 hash 결과와 무관하게 sentence 로 fallback.
결과적으로 데이터셋 시간은 보존 (utt 당 1 bin) 되면서 포맷 효과만 50/50 으로 분리됨.

word-interleaving 계획 배경 + 안 1/안 2 선택 근거: docs/word_alignment.md
  "Word-interleaving packing 계획" 섹션 참조.

사용법:
    # 단일 rank
    python precompute/half_inlv_pack_arrow.py --encoder fb_dacvae --rank 0 --num-ranks 8 --mixed

    # 8 rank 병렬 (wrapper 사용)
    bash precompute/run_half_inlv_pack.sh --encoder fb_dacvae

출력 경로:
    {precomputed_dir}/{encoder}/mixed/packed_half_inlv_{cutoff_len}/rank{N}_s{S}.arrow
    (기존 `packed_{cutoff_len}` 와 혼동 방지 위해 디렉토리 이름 분리)
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import random
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
    MergedAlignmentLookup,
    build_alignment_lookups,
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
# 1:1 split helper
# ======================================================================
def hash_split(utt_id) -> int:
    """deterministic 0/1. md5 기반 → Python 프로세스 salt 영향 없음.

    0 → sentence emit, 1 → interleave emit (alignment 있을 때).
    """
    if utt_id is None:
        return 0
    h = hashlib.md5(str(utt_id).encode("utf-8")).digest()
    return h[0] & 1


# ======================================================================
# processor_fn — utterance row → sentence OR interleave row (hash split)
# ======================================================================
def make_half_inlv_processor_fn(cfg, tokenizer,
                                 alignment_lookup: MergedAlignmentLookup | None,
                                 sentence_only: bool = False):
    """processor_fn 생성. 각 utterance 를 hash split 으로 1 row 씩 emit.

    출력 키 (row 당 1개):
        input_ids      : list<int>
        labels         : list<int>
        audio_features : list<float32>  — flat (T_enc × out_dim,)
        audio_lengths  : int            — T_enc (total, interleave 의 경우 zero-padded 합)

    interleave 포맷에서 projector stride 경계를 깔끔히 맞추기 위해 각 word 의
    feature 구간을 `total_stride` 배수로 end-zero-pad 한다 (truncate 가 아니라 pad —
    단어 음향 정보 손실 없음). 이렇게 하면 `ceil(sum T_wi_padded / stride)` 가
    `sum(T_wi_padded / stride)` 와 일치해서 input_ids 의 audio_pad 개수와 projector
    출력 frame 수가 정확히 같아짐 (±1 frame 방어 로직 불필요).

    Conv1d kernel (receptive field) 으로 인한 단어 경계 leak 는 수 frame 수준으로
    무시 가능하다고 판단. 문제가 생기면 word 간 zero-spacer 삽입으로 격리 가능.
    """
    audio_pad_id = cfg.get("audio_pad_token_id", 151655)
    max_text_len = cfg.get("max_text_len", 256)
    cutoff_len   = cfg.get("packing_cutoff_len", 16384)

    proj_strides = cfg["encoder"].get("proj_strides", [2, 2])
    total_stride = 1
    for s in proj_strides:
        total_stride *= s

    enc_cfg = cfg["encoder"]
    fps = enc_cfg["tgt_sr"] / enc_cfg["hop"]
    min_word_frames = max(2, int(0.1 * fps))   # ≥ 100ms per word

    eos_id = tokenizer.eos_token_id

    # §41 legacy text prompt 포맷 (model.py:AudioQwen 과 동일):
    #   input_ids = p1 + [audio_pad]*T_proj + p2 + text + [eos]
    # 이전 `<|audio_correspond|>` 단일 새 토큰 포맷이 수렴 실패의 주 원인
    # (docs/prompt_format_regression.md). 대체 텍스트 prompt 는 Qwen 이 이미 학습한
    # 일반 어휘라 LLM 의 language prior 를 anchor 로 사용 가능.
    p1_ids = tokenizer.encode("Audio:\n",        add_special_tokens=False)
    p2_ids = tokenizer.encode("\nTranscript:\n", add_special_tokens=False)
    len_p1 = len(p1_ids)
    len_p2 = len(p2_ids)

    def _emit_sentence(accum, flat_feats, T_enc, text):
        """sentence 포맷 한 row emit. legacy text prompt 사용."""
        if T_enc == 0:
            return
        text_ids = tokenizer.encode(text.lower().strip(), add_special_tokens=False)[:max_text_len]
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

    def _emit_interleave(accum, feats_2d, T_enc, out_dim, words) -> bool:
        """interleaved 한 row emit — 단어 하나마다 legacy text prompt 로 감쌈.

        포맷:
            p1 + [audio_pad]*T_w0 + p2 + text_w0 +
            p1 + [audio_pad]*T_w1 + p2 + text_w1 +
            ... + [eos]

        각 단어 청크가 legacy 포맷을 반복하는 형태라 LLM 은 매 단어마다
        anchoring 을 제공받음.
        """
        valid_word_count = 0
        input_ids_parts: list[int] = []
        labels_parts:    list[int] = []
        concat_feats_chunks: list[np.ndarray] = []
        total_padded_frames = 0

        for w in words:
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            start_frame = max(0, int(round(w.get("start", 0.0) * fps)))
            end_frame   = min(T_enc, int(round(w.get("end",   0.0) * fps)))
            if end_frame <= start_frame or (end_frame - start_frame) < min_word_frames:
                continue

            # 단어 feature 슬라이스
            word_feats = feats_2d[start_frame:end_frame]                          # (T_wi, out_dim)
            T_wi       = int(word_feats.shape[0])

            # total_stride 배수로 end-zero-pad
            pad_frames = (-T_wi) % total_stride
            if pad_frames:
                zero_pad = np.zeros((pad_frames, out_dim), dtype=np.float32)
                word_feats = np.concatenate([word_feats, zero_pad], axis=0)
            T_wi_padded = T_wi + pad_frames
            T_proj_wi   = T_wi_padded // total_stride
            if T_proj_wi == 0:
                continue

            # 단어 텍스트 토큰화
            word_text_ids = tokenizer.encode(
                word_text.lower().strip(), add_special_tokens=False
            )
            if not word_text_ids:
                continue

            # 청크 길이 = p1 + audio + p2 + word_text
            chunk_len = len_p1 + T_proj_wi + len_p2 + len(word_text_ids)
            if len(input_ids_parts) + chunk_len + 1 > cutoff_len:                 # +1 for EOS
                break

            input_ids_parts.extend(p1_ids)
            input_ids_parts.extend([audio_pad_id] * T_proj_wi)
            input_ids_parts.extend(p2_ids)
            input_ids_parts.extend(word_text_ids)

            labels_parts.extend([-100] * len_p1)
            labels_parts.extend([-100] * T_proj_wi)
            labels_parts.extend([-100] * len_p2)
            labels_parts.extend(word_text_ids)                                    # word text 만 supervise

            concat_feats_chunks.append(word_feats)
            total_padded_frames += T_wi_padded
            valid_word_count    += 1

        if valid_word_count < 2 or not input_ids_parts:
            return False

        input_ids_parts.append(eos_id)
        labels_parts.append(eos_id)

        if len(input_ids_parts) > cutoff_len:
            return False

        full_feats = np.concatenate(concat_feats_chunks, axis=0)                    # (T_total, out_dim)
        flat_feats = full_feats.flatten().tolist()

        accum["input_ids"].append(input_ids_parts)
        accum["labels"].append(labels_parts)
        accum["audio_features"].append(flat_feats)
        accum["audio_lengths"].append(int(total_padded_frames))
        return True

    def process_samples(examples):
        accum = {
            "input_ids":      [],
            "labels":         [],
            "audio_features": [],
            "audio_lengths":  [],
        }
        utt_ids = examples.get("utterance_id", [None] * len(examples["features"]))

        for utt_id, flat_feats, feat_len, text in zip(
            utt_ids, examples["features"], examples["feat_len"], examples["text"]
        ):
            if feat_len == 0 or not flat_feats:
                continue
            T_enc = int(feat_len)

            use_interleave = (
                (not sentence_only)
                and hash_split(utt_id) == 1
                and alignment_lookup is not None
                and utt_id is not None
            )

            emitted = False
            if use_interleave:
                words = alignment_lookup.get(utt_id)
                if words and len(words) >= 2:
                    arr = np.asarray(flat_feats, dtype=np.float32)
                    if arr.size % T_enc == 0:
                        out_dim = arr.size // T_enc
                        feats_2d = arr.reshape(T_enc, out_dim)
                        emitted = _emit_interleave(accum, feats_2d, T_enc, out_dim, words)

            if not emitted:
                _emit_sentence(accum, flat_feats, T_enc, text)

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
    per_ds = build_alignment_lookups(datasets_selected)
    alignment_lookup = MergedAlignmentLookup(list(per_ds.values()))
    _WORKER_PROCESSOR = make_half_inlv_processor_fn(
        cfg, tokenizer, alignment_lookup=alignment_lookup
    )


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

    out_dir = ds_dir / f"packed_half_inlv_{cutoff_len}"
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
    out_dir = base_dir / "mixed" / (pack_subdir or f"packed_half_inlv_{cutoff_len}")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"rank{rank}_s*.arrow"))
    if existing:
        print(f"  [skip] {len(existing)} shard(s) already exist for rank {rank}")
        return -1

    # Phase 1: 모든 데이터셋 processor_fn 통과 → temp Arrow 파일에 쓰기
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        suffix=f"_rank{rank}_half_inlv.arrow", dir="/mnt/tmp"
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
# Rank rebalance (§26 동일 로직, 출력 디렉토리만 packed_half_inlv_*)
# ======================================================================
def rebalance_mixed_shards(base_dir: Path, cutoff_len: int, num_ranks: int = 8,
                           pack_subdir: str | None = None):
    mixed_dir = base_dir / "mixed" / (pack_subdir or f"packed_half_inlv_{cutoff_len}")
    if not mixed_dir.exists():
        print(f"[rebalance] mixed dir not found: {mixed_dir}")
        return

    print(f"[rebalance] scanning {mixed_dir}")

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

    for r, (shards, total) in rank_info.items():
        excess = total - min_total
        if excess == 0:
            continue
        print(f"[rebalance] rank {r}: trim {excess:,} bins")

        for idx in range(len(shards) - 1, -1, -1):
            if excess <= 0:
                break
            shard_path, rows = shards[idx]
            if rows <= excess:
                shard_path.unlink()
                excess -= rows
                print(f"    rank {r}: removed shard {shard_path.name} ({rows} bins)")
            else:
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
    parser.add_argument("--sentence-only",   action="store_true",
                        help="§42: hash_split 결과 무시하고 모든 utt 를 sentence bin 으로 "
                             "패킹. interleave 비활성. 출력 디렉토리는 packed_sentence_only_{N}. "
                             "가설 A (interleave train/test mismatch) 검증용.")
    args = parser.parse_args()

    # §42: sentence-only 면 출력 subdir 이 달라져야 함. rebalance 도 해당 subdir 타겟.
    pack_subdir = (
        f"packed_sentence_only_{{cutoff_len}}" if args.sentence_only
        else f"packed_half_inlv_{{cutoff_len}}"
    )

    if args.rebalance:
        cfg_tmp = get_config(args.encoder)
        cutoff_len_tmp = args.cutoff_len or cfg_tmp["packing_cutoff_len"]
        base_dir_tmp = Path(args.precomputed_dir) / args.encoder
        rebalance_mixed_shards(
            base_dir_tmp, cutoff_len_tmp, num_ranks=args.num_ranks,
            pack_subdir=pack_subdir.format(cutoff_len=cutoff_len_tmp),
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
    print(f"Mode          : half interleave (utt-level hash split)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm_model"],
        cache_dir=cfg["model_cache_dir"],
        trust_remote_code=True,
    )

    # alignment lookup 은 interleave 분기 에만 필요하지만 utt-level split 결정
    # 이전에는 어떤 utt 가 interleave 로 가는지 모름 → 항상 빌드.
    selected = [k.strip() for k in args.datasets.split(",") if k.strip()]
    per_ds = build_alignment_lookups(selected)
    alignment_lookup = MergedAlignmentLookup(list(per_ds.values()))
    n_with_align = sum(1 for v in per_ds.values() if v is not None)
    print(f"AlignmentLookup: merged {n_with_align}/{len(per_ds)} datasets")
    if n_with_align == 0:
        print("  [warn] no alignment found — 모든 utt 가 sentence 로 fallback (안 2 효과 없음)")

    processor_fn = make_half_inlv_processor_fn(
        cfg, tokenizer, alignment_lookup=alignment_lookup,
        sentence_only=args.sentence_only,
    )
    print(f"Mode          : {'sentence-only (§42)' if args.sentence_only else 'half interleave'}")
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
            pack_subdir=pack_subdir.format(cutoff_len=cutoff_len),
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
