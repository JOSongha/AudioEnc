"""§43 In-place token-id rewrite for packed bins.

기존 packed bin (예: `mixed/packed_half_inlv_16384/`) 은 Qwen3.5 tokenizer 로 만들어져
id 범위가 Qwen3 (vocab 151,669) 을 넘어가므로 재사용 불가. 그러나 audio_features /
audio_lengths / bin packing 자체는 tokenizer 독립적이라, **각 bin 의 input_ids/labels
만 src_tok → dst_tok 로 swap** 해주면 offline 재패킹 없이 Qwen3 훈련에 재활용 가능.

알고리즘 (bin 별):
  1. attention_mask 가 0 이면 pad 영역, 실제 토큰 영역만 쪼개서 처리.
  2. 샘플 경계는 attention_mask 값 변화로 식별. 각 sample 내에서 chunk 를 3 종류로
     분류:
       A) audio_pad run (id == AUDIO_PAD=151655) — 그대로 복사
       B) non-audio-pad, labels == -100 — prompt (p1 / p2) 등. src_tok.decode →
          dst_tok.encode → 재삽입. labels = -100 유지.
       C) non-audio-pad, labels != -100 — text / eos. 동일하게 roundtrip. labels =
          new ids 그대로 복사.
     (chunk 경계는 (is_audio_pad, is_labeled) 튜플 변화 시점.)
  3. 재조립 후 길이 ≤ cutoff_len 이면 뒤를 dst_pad 로 padding + attention_mask=0.
     길이 초과 나는 경우는 현재 미지원 (Qwen3 tokenizer 가 Qwen3.5 보다 일반적으로
     더 짧거나 같아서 실제로는 발생 거의 없음 — 발생 시 assert).
  4. audio_features / audio_lengths 는 unchanged.

출력: `{src_dir}` → `{src_dir}_{dst_tag}` (예: `packed_half_inlv_16384` →
`packed_half_inlv_16384_qwen3_1.7b`). rank 단위 병렬 실행.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# config.py 는 AudioEnc root 에 있음 — 이 스크립트는 arXiv/scripts/precompute 하위
_AUDIOENC_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_AUDIOENC_ROOT))
from config import _infer_llm_family, _infer_llm_tag  # noqa: E402


AUDIO_PAD_ID = 151655   # Qwen3.5 / Qwen3 공통 (코드 하드코딩)


def _rewrite_row(
    input_ids: list[int],
    labels: list[int],
    attention_mask: list[int],
    src_tok,
    dst_tok,
    cutoff_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """단일 packed bin 의 input_ids/labels/attention_mask 를 dst_tok 기준으로 재작성.

    audio_features / audio_lengths 는 호출자가 그대로 보존.
    """
    dst_pad = dst_tok.pad_token_id if dst_tok.pad_token_id is not None else dst_tok.eos_token_id

    # attn==0 이 나오기 전까지 실제 token 범위.
    real_end = len(input_ids)
    for i, m in enumerate(attention_mask):
        if m == 0:
            real_end = i
            break

    new_ids: list[int] = []
    new_labels: list[int] = []
    new_attn: list[int] = []

    i = 0
    while i < real_end:
        att_i = attention_mask[i]     # 현재 sample index (>=1)
        # 한 sample 의 범위 찾기
        j = i
        while j < real_end and attention_mask[j] == att_i:
            j += 1
        # i..j : 한 sample

        # sample 내에서 chunk 경계 = (is_audio_pad, is_labeled) 변화
        k = i
        while k < j:
            is_audio = (input_ids[k] == AUDIO_PAD_ID)
            is_lbl   = (labels[k] != -100)
            m = k
            while m < j:
                if (input_ids[m] == AUDIO_PAD_ID) != is_audio:
                    break
                if not is_audio and (labels[m] != -100) != is_lbl:
                    break
                m += 1
            chunk_ids = input_ids[k:m]
            if is_audio:
                new_ids.extend(chunk_ids)                    # 그대로
                new_labels.extend([-100] * len(chunk_ids))
                new_attn.extend([att_i] * len(chunk_ids))
            else:
                # roundtrip via text — 특수 토큰 (eos 등) 포함 허용
                text   = src_tok.decode(chunk_ids, skip_special_tokens=False)
                re_ids = dst_tok.encode(text,       add_special_tokens=False)
                new_ids.extend(re_ids)
                if is_lbl:
                    new_labels.extend(list(re_ids))
                else:
                    new_labels.extend([-100] * len(re_ids))
                new_attn.extend([att_i] * len(re_ids))
            k = m
        i = j

    # Padding
    cur = len(new_ids)
    if cur > cutoff_len:
        raise ValueError(f"retokenized bin exceeds cutoff_len ({cur} > {cutoff_len}). "
                         "Qwen3 vocab 가 Qwen3.5 보다 드물게 더 많은 토큰을 요구하는 경계 샘플. "
                         "샘플 드롭 필요.")
    n_pad = cutoff_len - cur
    new_ids.extend([dst_pad]   * n_pad)
    new_labels.extend([-100]   * n_pad)
    new_attn.extend([0]        * n_pad)
    return new_ids, new_labels, new_attn


def rewrite_file(
    src_path: Path,
    dst_path: Path,
    src_tok,
    dst_tok,
    cutoff_len: int,
    verbose: bool = False,
    pbar_desc: str | None = None,
) -> tuple[int, int]:
    """단일 rank{N}_s{S}.arrow 파일 재작성. 반환: (rows, dropped).

    **streaming row-by-row**: batch 단위로 전체 column 을 메모리에 올리지 않고,
    각 row 를 slice(r, 1) 로 zero-copy 하게 떼어 재작성 → 즉시 1-row RecordBatch
    로 writer.write_batch() 호출. audio_features (거대한 list<list<float>>) 를
    한번에 materialize 하지 않음.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fld_ids   = None; fld_lab = None; fld_attn = None
    with pa.memory_map(str(src_path), "r") as mm:
        rdr = ipc.open_file(mm)
        schema = rdr.schema
        fld_ids  = schema.field("input_ids").type
        fld_lab  = schema.field("labels").type
        fld_attn = schema.field("attention_mask").type
        writer = ipc.new_file(str(dst_path), schema)
        n_rows  = 0
        n_drop  = 0
        total_bins = sum(rdr.get_batch(bi).num_rows for bi in range(rdr.num_record_batches))
        pbar = tqdm(total=total_bins, desc=pbar_desc or src_path.name,
                    unit="bin", dynamic_ncols=True)
        for bi in range(rdr.num_record_batches):
            batch = rdr.get_batch(bi)
            ids_col  = batch.column("input_ids")
            lab_col  = batch.column("labels")
            attn_col = batch.column("attention_mask")
            feat_col = batch.column("audio_features")
            alen_col = batch.column("audio_lengths")
            n = batch.num_rows
            for r in range(n):
                try:
                    ni, nl, na = _rewrite_row(
                        ids_col[r].as_py(),
                        lab_col[r].as_py(),
                        attn_col[r].as_py(),
                        src_tok, dst_tok, cutoff_len,
                    )
                except ValueError as e:
                    n_drop += 1
                    if verbose:
                        print(f"  [drop] row {r}: {e}", flush=True)
                    pbar.update(1)
                    continue
                # 1-row batch 구성. audio_features / audio_lengths 는 mmap-backed
                # pyarrow slice (zero-copy) 로 writer 가 직접 바이트 flush.
                one_batch = pa.RecordBatch.from_arrays(
                    [
                        pa.array([ni], type=fld_ids),
                        pa.array([nl], type=fld_lab),
                        pa.array([na], type=fld_attn),
                        feat_col.slice(r, 1),
                        alen_col.slice(r, 1),
                    ],
                    schema=schema,
                )
                writer.write_batch(one_batch)
                n_rows += 1
                pbar.update(1)
        pbar.close()
        writer.close()
    return n_rows, n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir",  required=True,
                    help="원본 packed dir (예: .../mixed/packed_half_inlv_16384)")
    ap.add_argument("--dst-dir",  default=None,
                    help="출력 dir. 생략 시 src_dir + suffix 자동 (예: "
                         "_qwen3_1.7b). suffix 는 --dst-llm 에서 유도.")
    ap.add_argument("--src-llm",  required=True,
                    help="원본 packed 를 만든 LLM (예: Qwen/Qwen3.5-2B)")
    ap.add_argument("--dst-llm",  required=True,
                    help="재작성 대상 LLM (예: Qwen/Qwen3-1.7B)")
    ap.add_argument("--cutoff-len", type=int, default=16384)
    ap.add_argument("--rank",     type=int, default=None,
                    help="지정된 rank{N}_*.arrow 만 처리 (병렬 실행용). 생략 시 전체.")
    ap.add_argument("--cache-dir", default="/mnt/tmp/cache/hf")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    # 출력 dir 기본 네이밍
    if args.dst_dir is None:
        fam = _infer_llm_family(args.dst_llm).lower().replace(".", "_")
        tag = _infer_llm_tag(args.dst_llm)
        suffix = f"_{fam}_{tag}"
        dst_dir = src_dir.parent / (src_dir.name + suffix)
    else:
        dst_dir = Path(args.dst_dir)

    print(f"[retokenize] src: {src_dir}")
    print(f"[retokenize] dst: {dst_dir}")
    print(f"[retokenize] src-llm: {args.src_llm}")
    print(f"[retokenize] dst-llm: {args.dst_llm}")
    print(f"[retokenize] cutoff_len: {args.cutoff_len}")

    src_tok = AutoTokenizer.from_pretrained(args.src_llm, cache_dir=args.cache_dir,
                                            trust_remote_code=True)
    dst_tok = AutoTokenizer.from_pretrained(args.dst_llm, cache_dir=args.cache_dir,
                                            trust_remote_code=True)

    # 대상 파일 리스트
    if args.rank is None:
        files = sorted(src_dir.glob("rank*_s*.arrow")) + sorted(src_dir.glob("rank*.arrow"))
        # 중복 제거
        files = sorted(set(files))
    else:
        files = sorted(src_dir.glob(f"rank{args.rank}_s*.arrow"))
        single = src_dir / f"rank{args.rank}.arrow"
        if single.exists() and single not in files:
            files.append(single)
    if not files:
        print(f"[retokenize] WARN: no files matched under {src_dir}")
        return

    total_rows = 0
    total_drop = 0
    for fp in files:
        dst_fp = dst_dir / fp.name
        # valid 한 기존 결과물만 스킵 — 0-byte / 중단된 파일은 재작성.
        if dst_fp.exists() and dst_fp.stat().st_size > 0:
            try:
                with pa.memory_map(str(dst_fp), "r") as _mm:
                    _rdr = ipc.open_file(_mm)
                    if _rdr.num_record_batches > 0:
                        print(f"[retokenize] skip (valid exists): {dst_fp}")
                        continue
            except Exception:
                pass   # fall through to rewrite
        print(f"[retokenize] {fp.name} → {dst_fp}", flush=True)
        rows, drop = rewrite_file(fp, dst_fp, src_tok, dst_tok, args.cutoff_len,
                                  verbose=args.verbose)
        total_rows += rows
        total_drop += drop
        print(f"  rows={rows} dropped={drop}", flush=True)
    print(f"[retokenize] done: {total_rows} rows, {total_drop} dropped")


if __name__ == "__main__":
    main()
