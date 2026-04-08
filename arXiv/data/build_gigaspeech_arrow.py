"""
GigaSpeech XL parquet → HuggingFace Arrow 변환.
parquet shard들이 /mnt/tmp/cache/hf/hub/datasets--speechcolab--gigaspeech/blobs/ 에 이미 있음.
Arrow 빌딩만 수행 (재다운로드 없음).

Usage:
    python3 build_gigaspeech_arrow.py \
        --cache-dir /mnt/tmp/cache \
        --subset    xl \
        --splits    train,validation,test \
        --num-proc  16
"""

import argparse
import os
import sys

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

HF_TOKEN = "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"
os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ.setdefault("HF_HOME", "/mnt/tmp/cache/hf")


def build(cache_dir: str, subset: str, splits: list[str], num_proc: int):
    from datasets import load_dataset, Audio

    # incomplete 디렉토리 정리
    import shutil, glob
    inc_pattern = os.path.join(
        cache_dir, "speechcolab___gigaspeech", subset, "0.0.0", "*.incomplete"
    )
    for p in glob.glob(inc_pattern):
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"removed incomplete dir: {p}")
        else:
            os.remove(p)
            print(f"removed incomplete file: {p}")

    for split in splits:
        print(f"\n[GigaSpeech {subset}/{split}] Arrow 빌딩 시작...", flush=True)
        try:
            ds = load_dataset(
                "speechcolab/gigaspeech",
                subset,
                split=split,
                cache_dir=cache_dir,
                token=HF_TOKEN,
                num_proc=num_proc,
                trust_remote_code=True,
            )
            print(f"[GigaSpeech {subset}/{split}] 완료: {len(ds):,}개", flush=True)
        except Exception as e:
            print(f"[GigaSpeech {subset}/{split}] 실패: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/mnt/tmp/cache")
    parser.add_argument("--subset",    default="xl")
    parser.add_argument("--splits",    default="train,validation,test")
    parser.add_argument("--num-proc",  default=16, type=int)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    build(args.cache_dir, args.subset, splits, args.num_proc)


if __name__ == "__main__":
    main()
