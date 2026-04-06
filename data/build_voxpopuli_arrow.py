"""
VoxPopuli English → HuggingFace Arrow 다운로드 및 변환.

Usage:
    python3 build_voxpopuli_arrow.py \
        --cache-dir /mnt/tmp/cache \
        --language  en \
        --splits    train,validation,test \
        --num-proc  8
"""

import argparse
import glob
import os
import sys

VENV = "/mnt/fr20tb/wbl_residency/jos/.venv310/lib/python3.10/site-packages"
if VENV not in sys.path:
    sys.path.insert(0, VENV)

HF_TOKEN = "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"
os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ.setdefault("HF_HOME", "/mnt/tmp/cache/hf")


def build(cache_dir: str, language: str, splits: list[str], num_proc: int):
    from datasets import load_dataset

    # 이전 incomplete blob 정리
    inc_pattern = os.path.join(
        "/mnt/tmp/cache/hf/hub/datasets--facebook--voxpopuli/blobs", "*.incomplete"
    )
    for p in glob.glob(inc_pattern):
        os.remove(p)
        print(f"removed incomplete blob: {p}")

    for split in splits:
        print(f"\n[VoxPopuli {language}/{split}] 다운로드 및 Arrow 빌딩 시작...", flush=True)
        try:
            ds = load_dataset(
                "facebook/voxpopuli",
                language,
                split=split,
                cache_dir=cache_dir,
                token=HF_TOKEN,
                num_proc=num_proc,
                trust_remote_code=True,
            )
            print(f"[VoxPopuli {language}/{split}] 완료: {len(ds):,}개", flush=True)
        except Exception as e:
            print(f"[VoxPopuli {language}/{split}] 실패: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/mnt/tmp/cache")
    parser.add_argument("--language",  default="en")
    parser.add_argument("--splits",    default="train,validation,test")
    parser.add_argument("--num-proc",  default=8, type=int)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    build(args.cache_dir, args.language, splits, args.num_proc)


if __name__ == "__main__":
    main()
