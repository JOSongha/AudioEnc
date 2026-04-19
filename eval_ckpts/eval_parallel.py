"""
eval_parallel.py — eval_all_ckpts.py 를 N GPU 에 분산 실행.

Strategy
--------
1. run-dir 에서 ckpt 목록 발견 (mtime sort).
2. ckpt 를 N GPU 에 round-robin 분배.
3. 각 GPU 에 subprocess 띄움: `eval_all_ckpts.py --only <subset> --gpu 0`
   (CUDA_VISIBLE_DEVICES 로 물리 GPU 격리 → 워커 안에서는 항상 cuda:0)
4. 모든 워커 종료 후 결과 디렉토리 의 모든 `<ckpt>.json` 을 모아 summary.json
   을 재구성 (ckpt 별로 따로 쓴 summary 들이 race 없이 합쳐짐).
5. build_viewer.py 자동 호출 → index.html 생성.

Usage
-----
    /mnt/ddn/users/jos/miniforge3/envs/audio/bin/python eval_ckpts/eval_parallel.py \
        --run-dir /mnt/tmp/cache/hf/fb_dacvae/s2_outputs_0414_1442 \
        --gpus 0,1,2,3 --split dev-clean --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_ckpts.eval_all_ckpts import discover_checkpoints  # noqa: E402

PYTHON = "/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python"


def partition(items: list, n: int) -> list[list]:
    """Round-robin partition so that mtime order is interleaved (균등 길이)."""
    out = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return out


def merge_summary(results_dir: Path, summary_template: dict) -> dict:
    """모든 워커가 쓴 per-ckpt JSON 을 모아 summary.json 재구성 (mtime sort)."""
    entries = []
    for p in results_dir.glob("*.json"):
        if p.name == "summary.json":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        entries.append({
            "ckpt": d["ckpt"],
            "saved_at": d.get("saved_at", ""),
            "saved_ts": d.get("saved_ts", 0),
            "wer": d["wer"],
            "n_samples": d["n_samples"],
            "elapsed_sec": d["elapsed_sec"],
            "json": p.name,
        })
    entries.sort(key=lambda e: e.get("saved_ts", 0))
    out = dict(summary_template)
    out["checkpoints"] = entries
    out["merged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=str, default="fb_dacvae")
    parser.add_argument("--split", type=str, default="dev-clean")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="특정 ckpt 만 (substring match)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="콤마 구분 GPU 번호 (e.g. 0,1,2,3)")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "eval_ckpts" / "results")
    parser.add_argument("--no-viewer", action="store_true",
                        help="build_viewer 자동 호출 생략")
    parser.add_argument("--stagger-sec", type=float, default=20.0,
                        help="워커 간 시작 간격 (PEFT wrap CUDA 경합 방지)")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    print(f"Using GPUs: {gpus}")

    ckpts = discover_checkpoints(args.run_dir)
    if args.only:
        ckpts = [c for c in ckpts if any(s in c.name for s in args.only)]
    if not ckpts:
        print("No checkpoints to evaluate.")
        return
    print(f"Total {len(ckpts)} ckpt(s); partitioning across {len(gpus)} GPU(s)")

    parts = partition(ckpts, len(gpus))
    for g, part in zip(gpus, parts):
        names = [p.name for p in part]
        print(f"  GPU {g}: {len(part)} ckpts → {', '.join(names)}")

    run_tag = args.run_dir.name
    out_dir = args.out_root / run_tag / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_logs"
    log_dir.mkdir(exist_ok=True)

    procs = []
    t0 = time.time()
    for g, part in zip(gpus, parts):
        if not part:
            continue
        only_args = ["--only", *[c.name for c in part]]
        cmd = [
            PYTHON, str(REPO_ROOT / "eval_ckpts" / "eval_all_ckpts.py"),
            "--run-dir", str(args.run_dir),
            "--encoder", args.encoder,
            "--split", args.split,
            "--max-samples", str(args.max_samples),
            "--max-new-tokens", str(args.max_new_tokens),
            "--beam-size", str(args.beam_size),
            "--seed", str(args.seed),
            "--gpu", "0",  # CUDA_VISIBLE_DEVICES 안에서는 0
            "--out-root", str(args.out_root),
            *only_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        log_path = log_dir / f"gpu{g}.log"
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(REPO_ROOT))
        procs.append((g, p, f, log_path, len(part)))
        print(f"  → launched GPU {g} pid {p.pid}, log: {log_path}")
        if args.stagger_sec > 0 and g != gpus[-1]:
            time.sleep(args.stagger_sec)

    # Wait for all
    print("\nWaiting for workers...")
    for g, p, f, log_path, n in procs:
        rc = p.wait()
        f.close()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  GPU {g} {status}  ({n} ckpts, {elapsed:.1f}s)  log: {log_path}")

    # Merge summary
    print("\nMerging summary.json ...")
    template = {
        "run_dir": str(args.run_dir),
        "encoder": args.encoder,
        "split": args.split,
        "n_samples": args.max_samples,
    }
    merged = merge_summary(out_dir, template)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    n_ck = len(merged["checkpoints"])
    print(f"  merged {n_ck} ckpts → {summary_path}")
    if n_ck:
        for e in merged["checkpoints"]:
            print(f"    {e['saved_at']}  {e['ckpt']:30s}  WER {e['wer']*100:6.2f}%")

    if not args.no_viewer:
        print("\nBuilding viewer ...")
        rc = subprocess.call([
            PYTHON, str(REPO_ROOT / "eval_ckpts" / "build_viewer.py"),
            str(out_dir),
        ])
        if rc == 0:
            print(f"  → file://{(out_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
