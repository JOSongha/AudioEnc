"""
Qwen3-ForcedAligner-0.6B를 이용한 word-level forced alignment.

대상: MLS / GigaSpeech / VoxPopuli
사용법:
    python3 align_qwen3.py \
        --dataset mls \
        --split   train \
        --gpu     0 \
        --rank    0 \
        --world   8 \
        --batch   16 \
        --out-dir /mnt/tmp/cache/word_alignments_qwen3

출력: {out-dir}/{dataset}/{split}/{utt_id}.json
"""

import argparse
import io
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

os.environ.setdefault("HF_HOME", "/mnt/tmp/cache/hf")
os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/tmp/cache/modelscope")
os.environ.setdefault("HF_TOKEN", "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")

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
MODEL_LOCAL = "/mnt/tmp/cache/modelscope/models/Qwen/Qwen3-ForcedAligner-0.6B"

# 데이터셋 설정
DATASET_CFG = {
    "mls": {
        "hf_name": "parler-tts/mls_eng_10k",
        "config":  None,
        "language": "English",
        "audio_field": "audio",
        "text_field":  "transcript",
        "id_field":    None,   # audio["path"] 에서 추출
        "decode_audio": False,
    },
    "gigaspeech": {
        "hf_name": "speechcolab/gigaspeech",
        "config":  "xl",
        "language": "English",
        "audio_field": "audio",
        "text_field":  "text",
        "id_field":    "segment_id",
        "decode_audio": False,
        "text_clean": lambda t: re.sub(r"<[^>]+>", "", t).strip(),
    },
    "voxpopuli": {
        "hf_name": "facebook/voxpopuli",
        "config":  "en",
        "language": "English",
        "audio_field": "audio",
        "text_field":  None,   # normalized_text or raw_text
        "id_field":    "audio_id",
        "decode_audio": False,
    },
}

TARGET_SR = 16000


def load_dataset_split(dataset: str, split: str, cache_dir: str):
    from datasets import load_dataset, Audio
    cfg = DATASET_CFG[dataset]
    kwargs = dict(split=split, cache_dir=cache_dir)
    if cfg["config"]:
        kwargs["name"] = cfg["config"]

    ds = load_dataset(cfg["hf_name"], **kwargs)

    # torchcodec(FFmpeg 의존) 회피: 항상 raw bytes로 읽고 torchaudio로 디코딩
    ds = ds.cast_column(cfg["audio_field"], Audio(decode=False))

    return ds


def get_utt_id(item, cfg, idx: int) -> str:
    if cfg["id_field"]:
        return str(item[cfg["id_field"]])
    # MLS: audio.path 에서 파일명 추출 (e.g. "6313/mls_eng_10k_6313_0.flac" → "6313_mls_eng_10k_6313_0")
    path = item["audio"].get("path", "") or ""
    name = Path(path).stem
    return name if name else str(idx)


def get_transcript(item, cfg) -> str:
    if cfg["text_field"]:
        text = item[cfg["text_field"]] or ""
    else:
        # VoxPopuli
        text = item.get("normalized_text") or item.get("raw_text") or ""
    clean_fn = cfg.get("text_clean")
    if clean_fn:
        text = clean_fn(text)
    return text.strip()


def load_audio_np(item, cfg) -> tuple:
    """(np.ndarray float32 mono, sample_rate) 반환"""
    audio_bytes = item["audio"]["bytes"]
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))

    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    return wav.numpy(), TARGET_SR


def align_batch(aligner, audio_list, text_list, language: str):
    """[(np, sr), ...] + [text, ...] → [ForcedAlignResult, ...]"""
    inputs = [(arr, sr) for arr, sr in audio_list]
    return aligner.align(inputs, text_list, [language] * len(inputs))


def save_json(out_path: Path, utt_id: str, dataset: str, split: str,
              audio_duration: float, transcript: str, result):
    words = []
    for item in result:
        words.append({
            "word":  item.text,
            "start": round(item.start_time, 4),
            "end":   round(item.end_time,   4),
        })
    record = {
        "utterance_id":    utt_id,
        "dataset":         dataset,
        "split":           split,
        "audio_duration":  round(audio_duration, 4),
        "transcript":      transcript,
        "words":           words,
        "alignment_model": MODEL_ID,
        "processed_at":    datetime.utcnow().isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, choices=list(DATASET_CFG.keys()))
    parser.add_argument("--split",     default="train")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--rank",      type=int, default=0)
    parser.add_argument("--world",     type=int, default=1)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--cache-dir", default="/mnt/tmp/cache")
    parser.add_argument("--out-dir",   default="/mnt/tmp/cache/word_alignments_qwen3")
    args = parser.parse_args()

    cfg = DATASET_CFG[args.dataset]
    device = f"cuda:{args.gpu}"

    # rank stagger (모델 로드 충돌 방지)
    if args.rank > 0:
        # 같은 GPU 내 워커끼리만 stagger (rank // world_per_gpu 순서)
        # rank % NUM_GPUS == gpu, 같은 GPU의 순서 = rank // NUM_GPUS
        # 단순하게: GPU 내 순서 × 15s
        local_order = args.rank // max(args.world // 8, 1)
        wait = min(local_order * 15, 120)
        if wait > 0:
            print(f"[rank {args.rank}] stagger {wait}s...", flush=True)
            time.sleep(wait)

    print(f"[rank {args.rank}] 모델 로드: {MODEL_LOCAL}", flush=True)
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained(
        MODEL_LOCAL,
        device_map=device,
        dtype=torch.bfloat16,
    )
    print(f"[rank {args.rank}] 모델 로드 완료", flush=True)

    print(f"[rank {args.rank}] 데이터셋 로드: {args.dataset}/{args.split}", flush=True)
    ds = load_dataset_split(args.dataset, args.split, args.cache_dir)
    total = len(ds)

    # rank 분할 (world 개 rank가 전체 데이터를 나눔)
    my_indices = list(range(args.rank, total, args.world))
    print(f"[rank {args.rank}] 담당: {len(my_indices):,}/{total:,}", flush=True)

    out_base = Path(args.out_dir) / args.dataset / args.split

    # Prefetch queue: producer(CPU decode) → consumer(GPU inference)
    prefetch_q = queue.Queue(maxsize=4)

    def producer():
        b_audio, b_text, b_meta = [], [], []
        for idx in my_indices:
            item = ds[idx]
            utt_id = get_utt_id(item, cfg, idx)
            out_path = out_base / f"{utt_id}.json"

            if out_path.exists():
                prefetch_q.put(('skip',))
                continue

            transcript = get_transcript(item, cfg)
            if not transcript:
                prefetch_q.put(('skip',))
                continue

            try:
                arr, sr = load_audio_np(item, cfg)
            except Exception as e:
                prefetch_q.put(('fail', utt_id, str(e)))
                continue

            dur = len(arr) / sr
            b_audio.append((arr, sr))
            b_text.append(transcript)
            b_meta.append((utt_id, dur, transcript, out_path))

            if len(b_audio) >= args.batch:
                prefetch_q.put(('batch', list(b_audio), list(b_text), list(b_meta)))
                b_audio.clear(); b_text.clear(); b_meta.clear()

        if b_audio:
            prefetch_q.put(('batch', b_audio, b_text, b_meta))
        prefetch_q.put(('done',))

    threading.Thread(target=producer, daemon=True).start()

    done = skip = fail = 0
    total_seen = last_log = 0
    t0 = time.time()

    while True:
        msg = prefetch_q.get()
        kind = msg[0]

        if kind == 'done':
            break
        elif kind == 'skip':
            skip += 1
            total_seen += 1
        elif kind == 'fail':
            fail += 1
            total_seen += 1
            print(f"[rank {args.rank}] 오디오 로드 실패 {msg[1]}: {msg[2]}", flush=True)
        elif kind == 'batch':
            _, b_audio, b_text, b_meta = msg
            try:
                results = align_batch(aligner, b_audio, b_text, cfg["language"])
                for res, (utt_id, dur, transcript, out_path) in zip(results, b_meta):
                    save_json(out_path, utt_id, args.dataset, args.split, dur, transcript, res)
                    done += 1
            except Exception as e:
                fail += len(b_audio)
                print(f"[rank {args.rank}] batch 실패: {e}", flush=True)
            total_seen += len(b_meta)

        if total_seen - last_log >= 200:
            last_log = total_seen
            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0
            print(f"[rank {args.rank}] {total_seen}/{len(my_indices)} | "
                  f"done={done} skip={skip} fail={fail} | "
                  f"{rate:.1f} utt/s", flush=True)

    elapsed = time.time() - t0
    print(f"[rank {args.rank}] 완료: done={done} skip={skip} fail={fail} "
          f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
