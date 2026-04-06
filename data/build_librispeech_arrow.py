"""
LibriSpeech raw flac → HuggingFace Arrow 변환.

flac bytes를 그대로 Arrow에 저장 (디코딩 없음).
출력 경로: {output_dir}/librispeech/{split}/

Usage:
    python3 build_librispeech_arrow.py \
        --librispeech-root /mnt/tmp/cache/LibriSpeech \
        --output-dir       /mnt/tmp/cache \
        --splits           train-clean-100,train-clean-360,train-other-500,dev-clean \
        --num-proc         16
"""

import argparse
import glob
import os
import sys

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

from datasets import Dataset, Features, Value
from tqdm import tqdm


def iter_split(root: str, split: str):
    """split 디렉토리에서 (utt_id, flac_path, transcript, speaker_id, chapter_id) yield."""
    split_dir = os.path.join(root, split)
    for trans_path in sorted(glob.glob(os.path.join(split_dir, "*", "*", "*.trans.txt"))):
        chapter_dir = os.path.dirname(trans_path)
        parts = chapter_dir[len(split_dir) + 1:].split("/")
        speaker_id, chapter_id = parts[0], parts[1]
        with open(trans_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, *rest = line.split()
                transcript = " ".join(rest).lower()
                flac_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                if not os.path.exists(flac_path):
                    continue
                yield utt_id, flac_path, transcript, speaker_id, chapter_id


def make_generator(root: str, split: str):
    def generator():
        for utt_id, flac_path, transcript, speaker_id, chapter_id in iter_split(root, split):
            with open(flac_path, "rb") as f:
                audio_bytes = f.read()
            yield {
                "audio":        {"bytes": audio_bytes, "path": flac_path},
                "transcript":   transcript,
                "utterance_id": utt_id,
                "speaker_id":   speaker_id,
                "chapter_id":   chapter_id,
                "split":        split,
            }
    return generator


# Audio()를 쓰면 torchcodec 인코딩 시도 → FFmpeg 없어서 실패.
# MLS arrow와 동일하게 struct<bytes: binary, path: string>으로 직접 정의.
FEATURES = Features({
    "audio":        {"bytes": Value("binary"), "path": Value("string")},
    "transcript":   Value("string"),
    "utterance_id": Value("string"),
    "speaker_id":   Value("string"),
    "chapter_id":   Value("string"),
    "split":        Value("string"),
})


def build_split(root: str, split: str, output_dir: str, num_proc: int, shard_size: int):
    out_path = os.path.join(output_dir, "librispeech", split)
    if os.path.exists(out_path):
        print(f"  [{split}] 이미 존재, skip. (삭제하려면 수동으로 제거)")
        return

    print(f"  [{split}] 카운트 중...", flush=True)
    entries = list(iter_split(root, split))
    print(f"  [{split}] {len(entries):,}개 utterance → Arrow 변환 시작", flush=True)

    gen = make_generator(root, split)
    ds = Dataset.from_generator(
        gen,
        features=FEATURES,
        num_proc=num_proc,
        writer_batch_size=shard_size,
    )

    print(f"  [{split}] save_to_disk: {out_path}", flush=True)
    ds.save_to_disk(out_path, num_proc=min(num_proc, 8), max_shard_size="500MB")
    print(f"  [{split}] 완료: {len(ds):,}개", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", default="/mnt/tmp/cache/LibriSpeech")
    parser.add_argument("--output-dir",       default="/mnt/tmp/cache")
    parser.add_argument("--splits",           default="train-clean-100,train-clean-360,train-other-500,dev-clean")
    parser.add_argument("--num-proc",         default=16, type=int)
    parser.add_argument("--shard-size",       default=500, type=int, help="writer_batch_size (utterances per shard)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    print(f"LibriSpeech → Arrow: {splits}")
    print(f"  root: {args.librispeech_root}")
    print(f"  out:  {args.output_dir}/librispeech/")
    print(f"  num_proc: {args.num_proc}")

    for split in splits:
        build_split(
            root=args.librispeech_root,
            split=split,
            output_dir=args.output_dir,
            num_proc=args.num_proc,
            shard_size=args.shard_size,
        )

    print("\n=== 전체 완료 ===")


if __name__ == "__main__":
    main()
