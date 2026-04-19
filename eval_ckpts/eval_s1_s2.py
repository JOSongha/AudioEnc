"""
eval_s1_s2.py — Stage 1 projector ckpts (.pt) 와 Stage 2 LoRA+projector ckpts
(checkpoint_fe*_split* 디렉토리) 를 한번에 돌려 WER 를 CSV 로 출력.

Stage 1 ckpt 는 projector+proj_norm 만 저장된 state_dict (~50 MB). 로드 시
AudioQwen 을 빌드 + apply_lora 없이 projector weight 만 덮어씀.

Stage 2 ckpt 는 디렉토리 안에 `model.safetensors` 가 있고 LoRA adapter + projector
가 섞여있음. eval_all_ckpts.load_eval_model 과 동일 경로.

모든 decoding 은 transcribe_train_format (§42 p1/p2 legacy prompt) 사용.

Usage:
    /mnt/ddn/users/jos/miniforge3/envs/audio/bin/python eval_ckpts/eval_s1_s2.py \\
        --s1-dir /mnt/tmp/cache/hf/fb_dacvae/s1_outputs_0415_1631 \\
        --s2-dir /mnt/tmp/cache/hf/fb_dacvae/s2_outputs_0415_1631 \\
        --split dev-clean --max-samples 200 --gpu 7 \\
        --out /tmp/s1_s2_wer.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from dataset import LibriSpeechDataset  # noqa: E402
from encoders import build_encoder  # noqa: E402
from inference import compute_wer  # noqa: E402
from model import AudioQwen  # noqa: E402
from eval_ckpts.eval_all_ckpts import transcribe_train_format  # noqa: E402


def _build_base_model(device: str, encoder_name: str = "fb_dacvae"):
    cfg = get_config(encoder_name)
    cfg = dict(cfg)
    cfg["attn_implementation"] = "sdpa"
    cfg["use_liger_kernel"] = False

    encoder = build_encoder(encoder_name, cfg["encoder"], cfg["model_cache_dir"])
    model = AudioQwen(encoder, cfg)
    return model, cfg


def load_s1_ckpt(pt_path: Path, device: str, encoder_name: str = "fb_dacvae"):
    """Stage 1 projector-only ckpt (.pt) 로드. LoRA 없음."""
    model, cfg = _build_base_model(device, encoder_name)
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # missing 은 LLM/encoder 전체 → 많음. unexpected 는 0 이 돼야 정상.
    if unexpected:
        print(f"  [s1] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device).eval()
    return model, cfg


def load_s2_ckpt(ckpt_dir: Path, device: str, encoder_name: str = "fb_dacvae"):
    """Stage 2 LoRA + projector ckpt (dir) 로드."""
    model, cfg = _build_base_model(device, encoder_name)
    model.apply_lora()

    sf_path = ckpt_dir / "model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(sf_path)
    from safetensors.torch import load_file
    state = load_file(str(sf_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [s2] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device).eval()
    return model, cfg


def discover_ckpts(s1_dir: Path | None, s2_dir: Path | None):
    """(stage, name, path, mtime) 튜플 리스트, mtime 오름차순."""
    items = []
    if s1_dir is not None and s1_dir.exists():
        for p in sorted(s1_dir.glob("s1_proj_split*.pt")):
            items.append(("s1", p.name, p, p.stat().st_mtime))
        final = s1_dir / "s1_proj.pt"
        if final.exists():
            items.append(("s1", final.name, final, final.stat().st_mtime))
    if s2_dir is not None and s2_dir.exists():
        for p in sorted(s2_dir.iterdir()):
            if p.is_dir() and (p / "model.safetensors").exists() and p.name.startswith("checkpoint_"):
                items.append(("s2", p.name, p, (p / "model.safetensors").stat().st_mtime))
    items.sort(key=lambda x: x[3])
    return items


def eval_one(model, cfg, dataset, indices, device, max_new_tokens=256):
    """returns (wer, refs, hyps)"""
    refs, hyps = [], []
    for i in tqdm(indices, leave=False):
        waveform, transcript = dataset[i]
        hyp = transcribe_train_format(waveform, model, cfg, device=device,
                                      max_new_tokens=max_new_tokens)
        refs.append(transcript)
        hyps.append(hyp)
    wer = compute_wer(hyps, refs)
    return wer, refs, hyps


CSV_FIELDS = ["order", "stage", "name", "wer_percent", "elapsed_sec", "mtime"]
PYTHON = "/mnt/ddn/users/jos/miniforge3/envs/audio/bin/python"


def _partition(items: list, n: int) -> list[list]:
    """Round-robin 분할: mtime 순서를 interleave 해서 GPU 부하 균등화."""
    out = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return out


def _save_ckpt_json(out_dir: Path, stage: str, name: str, mtime: float,
                    wer: float, refs: list, hyps: list, indices: list,
                    elapsed: float):
    """build_viewer 호환 schema 로 ckpt 당 JSON 저장."""
    data = {
        "ckpt": name,
        "stage": stage,
        "saved_at": datetime.fromtimestamp(mtime).strftime("%m/%d %H:%M:%S"),
        "saved_ts": mtime,
        "wer": wer,
        "n_samples": len(refs),
        "elapsed_sec": elapsed,
        "pairs": [{"idx": idx, "ref": r, "hyp": h}
                  for idx, r, h in zip(indices, refs, hyps)],
    }
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_sequential(args):
    """Single GPU 경로. CUDA_VISIBLE_DEVICES 는 shell/launcher 가 이미 설정했다고 가정.

    각 ckpt 당 `<out_dir>/<ckpt_name>.json` (build_viewer 호환) + `<out_dir>/summary.csv`
    (incremental) 을 저장. subprocess 내부에선 summary.csv 는 자기 subset 만 기록.
    """
    device = "cuda"
    s1_dir = Path(args.s1_dir) if args.s1_dir else None
    s2_dir = Path(args.s2_dir) if args.s2_dir else None

    cfg = get_config(args.encoder)
    dataset = LibriSpeechDataset(cache_dir=cfg["data_path"], url=args.split,
                                 max_len=cfg["max_audio_len"])
    n_total = len(dataset)
    # max_samples=0 이면 full
    n_eval = n_total if args.max_samples == 0 else min(args.max_samples, n_total)
    indices = list(range(n_eval))
    print(f"Eval on {args.split} ({n_eval}/{n_total} samples)")

    ckpts = discover_ckpts(s1_dir, s2_dir)
    # --only-names 필터
    if args.only_names:
        want = set(args.only_names)
        ckpts = [c for c in ckpts if c[1] in want]
    print(f"Evaluating {len(ckpts)} checkpoints "
          f"({sum(1 for c in ckpts if c[0]=='s1')} s1 + "
          f"{sum(1 for c in ckpts if c[0]=='s2')} s2)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    rows = []
    for order, (stage, name, path, mtime) in enumerate(ckpts, start=1):
        print(f"\n[{order}/{len(ckpts)}] {stage} {name}")
        t0 = time.time()
        try:
            if stage == "s1":
                model, mcfg = load_s1_ckpt(path, device, args.encoder)
            else:
                model, mcfg = load_s2_ckpt(path, device, args.encoder)
            with torch.inference_mode():
                wer, refs, hyps = eval_one(model, mcfg, dataset, indices, device,
                                           max_new_tokens=args.max_new_tokens)
            del model
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            wer_pct = wer * 100
            print(f"  WER = {wer_pct:.2f}%  ({elapsed:.1f}s)")
            _save_ckpt_json(out_dir, stage, name, mtime, wer, refs, hyps,
                            indices, elapsed)
            rows.append({"order": order, "stage": stage, "name": name,
                         "wer_percent": f"{wer_pct:.2f}",
                         "elapsed_sec": f"{elapsed:.1f}",
                         "mtime": time.strftime("%m-%d %H:%M:%S",
                                                time.localtime(mtime))})
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            rows.append({"order": order, "stage": stage, "name": name,
                         "wer_percent": "ERROR", "elapsed_sec": "",
                         "mtime": time.strftime("%m-%d %H:%M:%S",
                                                time.localtime(mtime))})

        # incremental write
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

    print(f"\nDone. {len(rows)} ckpts evaluated, CSV at {summary_csv}")


def _run_parallel(args, gpus: list[int]):
    """Multi GPU: ckpt 들을 GPU 수로 partition → subprocess 각각이 같은 out_dir 에
    per-ckpt JSON 저장 → 끝난 뒤 launcher 가 모든 JSON 모아 summary.json + CSV +
    build_viewer (HTML) 생성."""
    s1_dir = Path(args.s1_dir) if args.s1_dir else None
    s2_dir = Path(args.s2_dir) if args.s2_dir else None
    ckpts = discover_ckpts(s1_dir, s2_dir)
    if not ckpts:
        print("No checkpoints found.")
        return

    print(f"Launching parallel eval on GPUs {gpus} ({len(ckpts)} ckpts)")
    parts = _partition(ckpts, len(gpus))
    for g, part in zip(gpus, parts):
        names = [c[1] for c in part]
        print(f"  GPU {g}: {len(part)} ckpts → {', '.join(names)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root = out_dir / "_logs"
    log_root.mkdir(exist_ok=True)

    procs = []
    for g, part in zip(gpus, parts):
        if not part:
            continue
        cmd = [PYTHON, str(Path(__file__).resolve()),
               "--split", args.split,
               "--max-samples", str(args.max_samples),
               "--max-new-tokens", str(args.max_new_tokens),
               "--encoder", args.encoder,
               "--out", str(out_dir),
               "--only-names", *[c[1] for c in part]]
        if args.s1_dir:
            cmd += ["--s1-dir", args.s1_dir]
        if args.s2_dir:
            cmd += ["--s2-dir", args.s2_dir]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        log_path = log_root / f"gpu{g}.log"
        fh = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(Path(__file__).resolve().parent.parent))
        procs.append((g, p, fh, log_path, len(part)))
        print(f"  → launched GPU {g} pid {p.pid}, log: {log_path}")
        if args.stagger_sec > 0 and g != gpus[-1]:
            time.sleep(args.stagger_sec)

    t0 = time.time()
    print("\nWaiting for workers...")
    for g, p, fh, log_path, n in procs:
        rc = p.wait()
        fh.close()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  GPU {g} {status}  ({n} ckpts, {elapsed:.1f}s)  log: {log_path}")

    # subprocess 들이 per-ckpt JSON 을 out_dir 에 저장. 모아서 summary.json + CSV 생성.
    print("\nAggregating per-ckpt JSONs...")
    n_eval = None
    entries = []
    rows = []
    for order, (stage, name, path, mtime) in enumerate(ckpts, start=1):
        json_path = out_dir / f"{name}.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                d = json.load(f)
            if n_eval is None:
                n_eval = d["n_samples"]
            entries.append({
                "ckpt": d["ckpt"],
                "stage": d.get("stage", stage),
                "saved_at": d["saved_at"],
                "saved_ts": d["saved_ts"],
                "wer": d["wer"],
                "n_samples": d["n_samples"],
                "elapsed_sec": d["elapsed_sec"],
                "json": json_path.name,
            })
            rows.append({"order": order, "stage": stage, "name": name,
                         "wer_percent": f"{d['wer']*100:.2f}",
                         "elapsed_sec": f"{d['elapsed_sec']:.1f}",
                         "mtime": time.strftime("%m-%d %H:%M:%S",
                                                time.localtime(mtime))})
        else:
            print(f"  WARN: {json_path.name} missing")
            rows.append({"order": order, "stage": stage, "name": name,
                         "wer_percent": "MISSING", "elapsed_sec": "",
                         "mtime": time.strftime("%m-%d %H:%M:%S",
                                                time.localtime(mtime))})

    summary = {
        "s1_dir": args.s1_dir or "",
        "s2_dir": args.s2_dir or "",
        "encoder": args.encoder,
        "split": args.split,
        "n_samples": n_eval or args.max_samples,
        "checkpoints": entries,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"  summary.json: {out_dir/'summary.json'}")
    print(f"  summary.csv:  {out_dir/'summary.csv'}")

    # build_viewer → HTML
    viewer_script = Path(__file__).resolve().parent / "build_viewer.py"
    if viewer_script.exists():
        print("\nBuilding HTML viewer...")
        rc = subprocess.call([PYTHON, str(viewer_script), str(out_dir)])
        if rc == 0:
            print(f"  → file://{(out_dir / 'index.html').resolve()}")
        else:
            print(f"  build_viewer failed (rc={rc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-dir", type=str, default=None,
                    help="Stage 1 outputs dir (s1_proj*.pt 포함). 생략시 s1 eval skip.")
    ap.add_argument("--s2-dir", type=str, default=None,
                    help="Stage 2 outputs dir (checkpoint_fe*_split*/ 포함). 생략시 s2 eval skip.")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--encoder", default="fb_dacvae")
    ap.add_argument("--gpus", type=str, default=None,
                    help="콤마 구분 GPU 번호 (e.g. 0,1,2,3,4,5,6,7). 주면 parallel, "
                         "아니면 현재 visible GPU 로 sequential.")
    ap.add_argument("--only-names", nargs="*", default=None,
                    help="subprocess 내부 전용 — 이 이름들만 평가 (launcher 가 설정)")
    ap.add_argument("--stagger-sec", type=float, default=10.0,
                    help="Parallel 모드에서 worker 시작 간격 (AudioQwen 로드 CUDA 경합 완화)")
    ap.add_argument("--out", type=str, required=True,
                    help="output directory (per-ckpt JSON + summary.json + summary.csv + index.html)")
    args = ap.parse_args()

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        _run_parallel(args, gpus)
    else:
        _run_sequential(args)


if __name__ == "__main__":
    main()
