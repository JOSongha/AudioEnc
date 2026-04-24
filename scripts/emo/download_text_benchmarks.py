"""Download text MCQA benchmarks (ARC-easy/challenge, WinoGrande, HellaSwag,
BoolQ, PiQA, COPA) via HuggingFace datasets. Saves each split as a parquet
under /mnt/tmp/datasets/text_benchmarks/<name>/<split>.parquet and a summary
manifest.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from datasets import load_dataset

OUT = Path("/mnt/tmp/datasets/text_benchmarks")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("arc_easy",      "allenai/ai2_arc", "ARC-Easy"),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge"),
    ("winogrande",    "allenai/winogrande", "winogrande_xl"),
    ("hellaswag",     "Rowan/hellaswag", None),
    ("boolq",         "google/boolq", None),
    ("copa",          "aps/super_glue", "copa"),
    # PiQA dropped — `ybisk/piqa` legacy loader deprecated on HF; all alt mirrors 404.
]


def fetch(name: str, repo: str, config: str | None) -> dict:
    print(f"\n=== {name}  ({repo}{', ' + config if config else ''}) ===", flush=True)
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    try:
        ds = load_dataset(repo, config) if config else load_dataset(repo)
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        return {"_error": f"{type(e).__name__}: {e}"}
    for split in ds.keys():
        dst = out_dir / f"{split}.parquet"
        ds[split].to_parquet(dst)
        summary[split] = len(ds[split])
        print(f"   {split:<15} {len(ds[split]):>7}  → {dst}")
    return summary


def main() -> int:
    report: dict[str, dict] = {}
    for name, repo, config in TARGETS:
        try:
            report[name] = fetch(name, repo, config)
        except Exception:
            traceback.print_exc()
            report[name] = {"_exception": True}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2))
    print(f"\nsummary → {OUT / 'manifest.json'}")
    # quick total
    ok = {k: v for k, v in report.items() if "_error" not in v and "_exception" not in v}
    print(f"\nok: {len(ok)}/{len(TARGETS)}")
    for k, v in ok.items():
        print(f"  {k:<15} total={sum(v.values()):>7}  splits={list(v)}")
    return 0 if len(ok) == len(TARGETS) else 1


if __name__ == "__main__":
    sys.exit(main())
