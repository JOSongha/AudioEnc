"""
openslr/librispeech_asr → HuggingFace Arrow 다운로드 및 캐시.

config 'all': train.100, train.360, train.500, validation, test
출력: HF 기본 캐시 (/mnt/tmp/cache/hf/hub/datasets--openslr--librispeech_asr/)

Usage:
    python3 build_librispeech_hf_arrow.py \
        --cache-dir /mnt/tmp/cache \
        --config    all \
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

SPLITS = {
    "all":   ["train.clean.100", "train.clean.360", "train.other.500", "validation.clean", "validation.other", "test.clean", "test.other"],
    "clean": ["train.clean.100", "train.clean.360", "validation.clean", "test.clean"],
    "other": ["train.other.500", "validation.other", "test.other"],
}


def build(cache_dir: str, config: str, num_proc: int):
    from datasets import load_dataset

    splits = SPLITS.get(config, SPLITS["all"])
    for split in splits:
        print(f"\n[librispeech_asr {config}/{split}] 다운로드 및 Arrow 빌딩 시작...", flush=True)
        try:
            ds = load_dataset(
                "openslr/librispeech_asr",
                config,
                split=split,
                cache_dir=cache_dir,
                token=HF_TOKEN,
                num_proc=num_proc,
            )
            print(f"[librispeech_asr {config}/{split}] 완료: {len(ds):,}개", flush=True)
            print(f"  columns: {ds.column_names}", flush=True)
        except Exception as e:
            print(f"[librispeech_asr {config}/{split}] 실패: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/mnt/tmp/cache")
    parser.add_argument("--config",    default="all", choices=["all", "clean", "other"])
    parser.add_argument("--num-proc",  default=16, type=int)
    args = parser.parse_args()

    build(args.cache_dir, args.config, args.num_proc)


if __name__ == "__main__":
    main()
