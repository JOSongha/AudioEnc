"""
단일 GPU 워커: 할당된 shard를 wav2vec2-large CTC forced alignment로 처리.
flac 디코딩을 ThreadPoolExecutor로 prefetch → GPU 연산과 I/O 오버랩.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 align_worker.py \
        --shard /mnt/tmp/cache/word_alignments/shard_0.jsonl \
        --output-dir /mnt/tmp/cache/word_alignments \
        --gpu 0 \
        --rank 0
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

import whisperx

MODEL_NAME = "WAV2VEC2_ASR_LARGE_LV60K_960H"
SAMPLE_RATE = 16000
PREFETCH = 4  # GPU 연산 중 미리 로드할 flac 수


def load_flac(flac_path: str) -> np.ndarray:
    """flac → 16kHz mono numpy float32"""
    waveform, sr = torchaudio.load(flac_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


def output_path(output_dir: str, entry: dict) -> str:
    return os.path.join(
        output_dir,
        entry["split"],
        entry["speaker_id"],
        entry["chapter_id"],
        entry["utterance_id"] + ".json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",      required=True)
    parser.add_argument("--output-dir", default="/mnt/tmp/cache/word_alignments")
    parser.add_argument("--gpu",        default=0, type=int)
    parser.add_argument("--rank",       default=0, type=int,
                        help="0번 rank는 모델 다운로드 후 진행, 나머지는 10초 대기")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # rank-0이 먼저 모델 다운로드 → 나머지는 캐시에서 로드
    # 같은 GPU 내 여러 워커: rank * 2초 stagger로 GPU 메모리 할당 충돌 방지
    if args.rank != 0:
        wait_sec = min(args.rank * 2, 30)   # rank-1=2s, rank-2=4s, ..., max 30s
        print(f"[rank {args.rank}] stagger 대기 ({wait_sec}s)...")
        time.sleep(wait_sec)

    print(f"[rank {args.rank}] Loading alignment model: {MODEL_NAME} on {device}")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en",
        device=device,
        model_name=MODEL_NAME,
    )
    print(f"[rank {args.rank}] Model loaded.")

    # shard 읽기
    with open(args.shard) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    total = len(entries)
    done = skipped = failed = 0

    log_path = os.path.join(args.output_dir, f"worker_{args.rank}.log")
    log_f = open(log_path, "a")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}][rank {args.rank}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"Start: {total} utterances (prefetch={PREFETCH})")

    # prefetch 큐: (entry, Future[ndarray] | None)
    # None이면 스킵 대상 (출력 파일 이미 존재)
    queue: list[tuple[dict, Optional[Future]]] = []

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        def submit_next(idx: int):
            if idx >= total:
                return None
            e = entries[idx]
            if os.path.exists(output_path(args.output_dir, e)):
                return (e, None)          # 스킵
            return (e, pool.submit(load_flac, e["flac_path"]))

        # 첫 PREFETCH개 제출
        fetch_idx = 0
        while len(queue) < PREFETCH and fetch_idx < total:
            item = submit_next(fetch_idx)
            if item is not None:
                queue.append(item)
            fetch_idx += 1

        pbar = tqdm(total=total, desc=f"rank{args.rank}", unit="utt",
                    dynamic_ncols=True, position=args.rank)

        i = 0
        while queue:
            entry, fut = queue.pop(0)

            # 다음 항목 prefetch
            while len(queue) < PREFETCH and fetch_idx < total:
                item = submit_next(fetch_idx)
                if item is not None:
                    queue.append(item)
                fetch_idx += 1

            # 스킵
            if fut is None:
                skipped += 1
                i += 1
                pbar.update(1)
                pbar.set_postfix(done=done, skip=skipped, fail=failed)
                continue

            try:
                audio = fut.result()          # 이미 CPU에서 로드 완료
                duration = len(audio) / SAMPLE_RATE
                out_path = output_path(args.output_dir, entry)

                transcript_segments = [{
                    "text":  entry["transcript"],
                    "start": 0.0,
                    "end":   duration,
                }]

                result = whisperx.align(
                    transcript_segments,
                    align_model,
                    align_metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )

                words = []
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        words.append({
                            "word":  w.get("word", "").strip(),
                            "start": round(w.get("start", 0.0), 4),
                            "end":   round(w.get("end",   0.0), 4),
                            "score": round(float(w.get("score", 0.0)), 4),
                        })

                out = {
                    "utterance_id":    entry["utterance_id"],
                    "split":           entry["split"],
                    "speaker_id":      entry["speaker_id"],
                    "chapter_id":      entry["chapter_id"],
                    "audio_duration":  round(duration, 4),
                    "transcript":      entry["transcript"],
                    "words":           words,
                    "alignment_model": MODEL_NAME,
                    "processed_at":    datetime.now(timezone.utc).isoformat(),
                }

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(out, f, ensure_ascii=False)

                done += 1

            except Exception as e:
                failed += 1
                log(f"FAIL {entry['utterance_id']}: {e}")

            i += 1
            pbar.update(1)
            pbar.set_postfix(done=done, skip=skipped, fail=failed)
            if i % 500 == 0:
                log(f"Progress: {i}/{total} | done={done} skip={skipped} fail={failed}")

        pbar.close()

    log(f"Done: total={total} done={done} skipped={skipped} failed={failed}")
    log_f.close()


if __name__ == "__main__":
    main()
