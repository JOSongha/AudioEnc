#!/usr/bin/env python3
"""
precompute_features.py — 인코더 출력 피처를 사전 계산해 Arrow 파일로 저장.

단일 GPU 프로세스. run_precompute.sh가 8 GPU를 병렬로 실행한다.

출력 경로:
    {output_dir}/{encoder_name}/{dataset}/rank{gpu_id}.arrow
    {output_dir}/{encoder_name}/meta.json

Arrow 스키마:
    utterance_id  : string
    text          : string
    features      : list<float32>  — shape (feat_len × out_dim,), flattened
    feat_len      : int32          — 인코더 출력 프레임 수

사용법:
    # 8 GPU 병렬 (권장):
    bash precompute/run_precompute.sh --encoder fb_dacvae

    # 단일 GPU 직접:
    CUDA_VISIBLE_DEVICES=0 python precompute/precompute_features.py \\
        --encoder fb_dacvae --gpu-id 0 --num-gpus 8

    # 특정 데이터셋만:
    CUDA_VISIBLE_DEVICES=0 python precompute/precompute_features.py \\
        --encoder fb_dacvae --gpu-id 0 --num-gpus 8 --datasets ls100,gs

    # 완료 확인 (피처 수, 파일 크기):
    python precompute/precompute_features.py --encoder fb_dacvae --verify
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf
import torch
import torchaudio.functional as AF

# AudioEnc 루트를 경로에 추가 (encoders/, config.py 임포트용)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import ENCODER_REGISTRY, get_config
from encoders import ENCODER_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "/mnt/fr20tb/wbl_residency/jos/ddn/precomputed"
HF_CACHE_DIR       = "/mnt/tmp/cache"
HF_MODEL_CACHE     = "/mnt/tmp/cache/hf"
SRC_SR             = 16000   # 모든 HF 데이터셋 오디오는 16kHz (decode 후)

# 데이터셋별 utterance_id 필드 및 텍스트 필드
_DATASET_FIELDS = {
    "ls100": ("id",            None,                         "text"),
    "ls360": ("id",            None,                         "text"),
    "ls500": ("id",            None,                         "text"),
    "mls":   ("original_path", lambda s: s.removesuffix(".opus"), "transcript"),
    "gs":    ("segment_id",    None,                         "text"),
    "vp":    ("audio_id",      None,                         "normalized_text"),
}

# HF 데이터셋 로드 인자
_DATASET_LOAD_ARGS = {
    "ls100": dict(path="openslr/librispeech_asr", name=None,   split="train.clean.100"),
    "ls360": dict(path="openslr/librispeech_asr", name=None,   split="train.clean.360"),
    "ls500": dict(path="openslr/librispeech_asr", name=None,   split="train.other.500"),
    "mls":   dict(path="parler-tts/mls_eng_10k",  name=None,   split="train"),
    "gs":    dict(path="speechcolab/gigaspeech",  name="xl",   split="train"),
    "vp":    dict(path="facebook/voxpopuli",      name="en",   split="train"),
}

# Arrow 스키마
_SCHEMA = pa.schema([
    pa.field("utterance_id", pa.string()),
    pa.field("text",         pa.string()),
    pa.field("features",     pa.list_(pa.float32())),
    pa.field("feat_len",     pa.int32()),
])

# GPU당 동시 처리 배치 수 (CUDA stream 파이프라인)
N_STREAMS = 2

# CPU 오디오 디코딩 스레드 수
N_DECODE_THREADS = 16

# 한 번에 GPU에 올리는 최대 클립 수 (A100 80GB 기준)
DEFAULT_BATCH_SIZE = 16

# Arrow 파일 flush 주기 (row 수)
FLUSH_EVERY = 2000


# ──────────────────────────────────────────────────────────────────────────────
# 오디오 디코딩
# ──────────────────────────────────────────────────────────────────────────────

def _decode_audio(audio_bytes: bytes, target_sr: int = SRC_SR) -> Optional[np.ndarray]:
    """오디오 bytes → float32 numpy (1D, mono). 실패 시 None."""
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            t = torch.from_numpy(wav).unsqueeze(0)
            t = AF.resample(t, sr, target_sr)
            wav = t.squeeze(0).numpy()
        return wav
    except Exception as e:
        print(f"[warn] decode error: {e}", flush=True)
        return None


def _decode_batch_parallel(raw_items: list[dict], id_field: str, id_transform, text_field: str,
                            executor: ThreadPoolExecutor) -> list[dict]:
    """raw HF 항목 리스트를 병렬 디코딩. [{id, text, wav}, ...] 반환."""
    def _decode_one(item):
        audio_bytes = item["audio"]["bytes"]
        wav = _decode_audio(audio_bytes)
        if wav is None:
            return None
        uid = item[id_field]
        if id_transform:
            uid = id_transform(uid)
        text = item.get(text_field, "").strip()
        return {"id": uid, "text": text, "wav": wav}

    futures = [executor.submit(_decode_one, item) for item in raw_items]
    results = []
    for f in futures:
        r = f.result()
        if r is not None:
            results.append(r)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 인코더 forward (배치, CUDA stream)
# ──────────────────────────────────────────────────────────────────────────────

def _encode_batch_impl(encoder, wavs: list[np.ndarray], device: torch.device,
                       stream: torch.cuda.Stream) -> list[np.ndarray]:
    """wav 리스트 → 인코더 피처 리스트. 내부 구현 (OOM fallback에서 단건 호출도 함)."""
    lengths = [len(w) for w in wavs]
    max_len = max(lengths)
    padded = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        padded[i, :len(w)] = w

    with torch.cuda.stream(stream):
        wav_t = torch.from_numpy(padded).to(device, non_blocking=True)
        len_t = torch.tensor(lengths, dtype=torch.long, device=device)

        with torch.no_grad():
            feats, enc_mask = encoder(wav_t, len_t)   # (B, T_enc, out_dim)

        feat_lens = enc_mask.sum(dim=1).cpu().tolist()
        feats_np = feats.float().cpu().numpy()

    results = []
    for i, fl in enumerate(feat_lens):
        results.append(feats_np[i, :int(fl), :])
    return results


def _encode_batch(encoder, wavs: list[np.ndarray], device: torch.device,
                  stream: torch.cuda.Stream) -> list[np.ndarray]:
    """wav 리스트 → 인코더 피처 리스트. OOM 시 clip-by-clip fallback."""
    if not wavs:
        return []
    try:
        return _encode_batch_impl(encoder, wavs, device, stream)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[warn] OOM on batch size {len(wavs)}, falling back to clip-by-clip", flush=True)
        results = []
        for wav in wavs:
            try:
                r = _encode_batch_impl(encoder, [wav], device, stream)
                results.extend(r)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[warn] OOM on single clip (len={len(wav)}), skipping", flush=True)
                # 빈 피처로 채워서 downstream에서 skipped 처리
                results.append(np.zeros((0, encoder.out_dim), dtype=np.float32))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋 로딩
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset_shard(dataset_key: str, gpu_id: int, num_gpus: int):
    """HF 스트리밍 데이터셋 로드 후 이 GPU 샤드만 반환."""
    from datasets import Audio, load_dataset

    args = _DATASET_LOAD_ARGS[dataset_key]
    kwargs = dict(
        path=args["path"],
        split=args["split"],
        streaming=True,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
    )
    if args["name"]:
        kwargs["name"] = args["name"]

    ds = load_dataset(**kwargs)

    if num_gpus > 1:
        ds = ds.shard(num_shards=num_gpus, index=gpu_id, contiguous=False)

    # 오디오는 bytes로 유지 (decode=False → dict with "bytes", "path")
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 메인 전처리 루프 (단일 GPU, 단일 데이터셋)
# ──────────────────────────────────────────────────────────────────────────────

def precompute_dataset(
    encoder,
    dataset_key: str,
    gpu_id: int,
    num_gpus: int,
    output_path: Path,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """데이터셋 한 개의 이 GPU 샤드를 전처리해 Arrow 파일로 저장."""
    id_field, id_transform, text_field = _DATASET_FIELDS[dataset_key]

    print(f"[GPU {gpu_id}] {dataset_key}: loading shard {gpu_id}/{num_gpus} ...", flush=True)
    ds = _load_dataset_shard(dataset_key, gpu_id, num_gpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".arrow.tmp")
    tmp_path.unlink(missing_ok=True)
    writer = ipc.new_file(str(tmp_path), _SCHEMA)

    # CUDA 스트림 × N_STREAMS (파이프라인)
    streams = [torch.cuda.Stream(device=device) for _ in range(N_STREAMS)]

    # CPU 디코딩 스레드 풀
    executor = ThreadPoolExecutor(max_workers=N_DECODE_THREADS)

    # 누적 버퍼 (Arrow flush 전까지)
    buf_ids, buf_texts, buf_feats, buf_lens = [], [], [], []

    total_written = 0
    total_skipped = 0
    t0 = time.time()

    # 스트리밍 배치 처리
    raw_buf = []
    stream_idx = 0

    def _flush_arrow():
        nonlocal buf_ids, buf_texts, buf_feats, buf_lens
        if not buf_ids:
            return
        batch = pa.record_batch({
            "utterance_id": pa.array(buf_ids,  type=pa.string()),
            "text":         pa.array(buf_texts, type=pa.string()),
            "features":     pa.array([f.flatten().tolist() for f in buf_feats],
                                     type=pa.list_(pa.float32())),
            "feat_len":     pa.array(buf_lens,  type=pa.int32()),
        }, schema=_SCHEMA)
        writer.write_batch(batch)
        buf_ids, buf_texts, buf_feats, buf_lens = [], [], [], []

    def _process_raw(raw_items):
        nonlocal total_written, total_skipped, stream_idx
        decoded = _decode_batch_parallel(raw_items, id_field, id_transform,
                                         text_field, executor)
        if not decoded:
            return

        wavs  = [d["wav"]  for d in decoded]
        ids   = [d["id"]   for d in decoded]
        texts = [d["text"] for d in decoded]

        feats = _encode_batch(encoder, wavs, device, streams[stream_idx % N_STREAMS])
        stream_idx += 1

        for uid, text, feat in zip(ids, texts, feats):
            if feat.shape[0] == 0:
                total_skipped += 1
                continue
            buf_ids.append(uid)
            buf_texts.append(text)
            buf_feats.append(feat)
            buf_lens.append(feat.shape[0])
            total_written += 1

        if total_written % FLUSH_EVERY < batch_size:
            _flush_arrow()

    success = False
    try:
        for item in ds:
            raw_buf.append(item)
            if len(raw_buf) >= batch_size:
                _process_raw(raw_buf)
                raw_buf = []

                elapsed = time.time() - t0
                rate = total_written / elapsed if elapsed > 0 else 0
                print(
                    f"[GPU {gpu_id}] {dataset_key}: "
                    f"written={total_written:,}  skip={total_skipped}  "
                    f"{rate:.1f} utt/s",
                    flush=True,
                )

        # 나머지
        if raw_buf:
            _process_raw(raw_buf)

        success = True

    finally:
        # 남은 버퍼 flush & 파일 닫기
        _flush_arrow()
        writer.close()
        executor.shutdown(wait=True)
        if success:
            tmp_path.rename(output_path)   # 완료된 경우에만 최종 경로로 이동
        else:
            tmp_path.unlink(missing_ok=True)  # 실패 시 임시 파일 삭제

    elapsed = time.time() - t0
    print(
        f"[GPU {gpu_id}] {dataset_key}: DONE  "
        f"written={total_written:,}  skip={total_skipped}  "
        f"elapsed={elapsed/3600:.2f}h  path={output_path}",
        flush=True,
    )
    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# 완료 확인 (--verify)
# ──────────────────────────────────────────────────────────────────────────────

def verify(encoder_name: str, output_dir: Path):
    meta_path = output_dir / encoder_name / "meta.json"
    if not meta_path.exists():
        print(f"meta.json 없음: {meta_path}")
        return

    meta = json.loads(meta_path.read_text())
    total = 0
    for ds_key, files in meta.get("files", {}).items():
        for fpath, n_rows in files.items():
            exists = Path(fpath).exists()
            size_mb = Path(fpath).stat().st_size / 1e6 if exists else 0
            status = "OK" if exists else "MISSING"
            print(f"  [{status}] {fpath}  rows={n_rows:,}  {size_mb:.0f} MB")
            total += n_rows
    print(f"\n총 utterance: {total:,}")


# ──────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute encoder features")
    parser.add_argument("--encoder",    required=True,
                        choices=list(ENCODER_REGISTRY),
                        help="인코더 이름")
    parser.add_argument("--gpu-id",     type=int, default=0,
                        help="이 프로세스가 사용할 GPU 인덱스 (CUDA_VISIBLE_DEVICES=0 → gpu-id=0)")
    parser.add_argument("--num-gpus",   type=int, default=8,
                        help="병렬 GPU 총 수 (샤딩에 사용)")
    parser.add_argument("--datasets",   default="all",
                        help="처리할 데이터셋 (쉼표 구분). 'all' = ls100,ls360,ls500,mls,gs,vp")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"GPU 배치 크기 (기본 {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"출력 디렉토리 (기본 {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--cache-dir",  default=HF_CACHE_DIR,
                        help="HuggingFace 데이터 캐시")
    parser.add_argument("--model-cache-dir", default=HF_MODEL_CACHE,
                        help="HuggingFace 모델 캐시")
    parser.add_argument("--verify",     action="store_true",
                        help="전처리 결과 확인만 하고 종료")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.verify:
        verify(args.encoder, output_dir)
        return

    # 데이터셋 목록
    if args.datasets == "all":
        selected = ["ls100", "ls360", "ls500", "mls", "gs", "vp"]
    else:
        selected = [d.strip() for d in args.datasets.split(",")]

    # GPU 설정
    device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES로 GPU 선택됨
    torch.cuda.set_device(device)

    # 인코더 로드
    cfg = get_config(args.encoder)
    cfg["model_cache_dir"] = args.model_cache_dir

    print(f"[GPU {args.gpu_id}] Loading encoder '{args.encoder}' ...", flush=True)
    EncoderClass = ENCODER_CLASSES[args.encoder]
    encoder = EncoderClass(cfg["encoder"], cache_dir=args.model_cache_dir)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[GPU {args.gpu_id}] Encoder loaded. out_dim={encoder.out_dim}", flush=True)

    # 메타 정보
    meta = {
        "encoder": args.encoder,
        "out_dim": encoder.out_dim,
        "tgt_sr":  cfg["encoder"]["tgt_sr"],
        "hop":     cfg["encoder"]["hop"],
        "files":   {},
    }

    # 각 데이터셋 전처리
    for ds_key in selected:
        out_path = output_dir / args.encoder / ds_key / f"rank{args.gpu_id}.arrow"

        if out_path.exists():
            print(f"[GPU {args.gpu_id}] {ds_key}: already exists, skipping. "
                  f"(삭제 후 재실행하려면 {out_path})", flush=True)
            continue

        n = precompute_dataset(
            encoder=encoder,
            dataset_key=ds_key,
            gpu_id=args.gpu_id,
            num_gpus=args.num_gpus,
            output_path=out_path,
            device=device,
            batch_size=args.batch_size,
        )
        meta["files"].setdefault(ds_key, {})[str(out_path)] = n

    # meta.json 저장 (rank 0만)
    if args.gpu_id == 0:
        meta_path = output_dir / args.encoder / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        # 기존 meta가 있으면 merge
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            for ds_key, files in existing.get("files", {}).items():
                meta["files"].setdefault(ds_key, {}).update(files)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[GPU 0] meta.json 저장: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
