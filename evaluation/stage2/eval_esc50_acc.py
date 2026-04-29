"""ESC-50 5-fold accuracy eval for Qwen3.5AE Stage-2 checkpoints.

Prompt: TASK_PROMPTS["sound_classify_single"][0] — matches training single-
class captioning. Target: a single class label from the 50-class vocabulary
(training stores raw labels like "dog", "chainsaw"; we compare normalized).

5-fold CV convention: each row carries a `fold` ∈ {1..5}. Standard ESC-50
eval protocol holds ONE fold out at a time; with --fold N we eval only fold N;
with --all we eval all 5 and report per-fold + mean accuracy.

Usage:
    python -m evaluation.stage2.eval_esc50_acc \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_esc50 \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 2000 --all --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.stage2._loader import (  # noqa: E402
    audio_sample_rate,
    build_prompt_ids,
    default_max_audio_samples,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
    t_audio_for,
)

ESC_ROOT = Path("/mnt/tmp/datasets/env_sound/ESC-50")
META_CSV = ESC_ROOT / "meta/esc50.csv"
AUDIO_DIR = ESC_ROOT / "audio"
EVAL_STEM = "Classify this sound."
MAX_NEW_TOKENS = 32


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_meta() -> list[dict]:
    rows = []
    classes: set[str] = set()
    with open(META_CSV) as f:
        for r in csv.DictReader(f):
            ap = AUDIO_DIR / r["filename"]
            if not ap.exists():
                continue
            rows.append({
                "filename": r["filename"],
                "path": str(ap),
                "fold": int(r["fold"]),
                "label": r["category"],  # e.g. "chainsaw", "rooster"
            })
            classes.add(r["category"])
    return rows, sorted(classes)


def preprocess_audio(path: str, target_sr: int, max_samples: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if wav.shape[-1] > max_samples:
        wav = wav[:max_samples]
    return wav


_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_label(s: str) -> str:
    return _NORM_RE.sub("_", s.strip().lower()).strip("_")


def parse_single_label(text: str, classes_norm: dict[str, str]) -> str | None:
    """Return canonical class name (raw, e.g. 'chainsaw') or None.

    Resolution order:
        1. Exact-normalized match.
        2. Class key as substring of prediction ("dog bark" -> "dog").
        3. Prediction as substring of class key ("cough" -> "coughing").
        4. First whitespace/underscore token match.
    """
    head = text.split("\n")[0].split(".")[0].strip()
    head_norm = norm_label(head)
    if not head_norm:
        return None
    if head_norm in classes_norm:
        return classes_norm[head_norm]
    for key, canon in classes_norm.items():
        if key and key in head_norm:
            return canon
    for key, canon in classes_norm.items():
        if key and head_norm in key:
            return canon
    first = head_norm.split("_")[0]
    if first and first in classes_norm:
        return classes_norm[first]
    return None


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_batch(model, tokenizer, cfg, batch: list[dict],
              max_new_tokens: int, use_cache: bool = True) -> list[str]:
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in batch:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, EVAL_STEM))
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
    )


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    rows: list[dict],
    classes: list[str],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int,
    max_new_tokens: int,
    folds_to_eval: list[int],
    use_cache: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    target_sr = audio_sample_rate(cfg)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)
    classes_norm = {norm_label(c): c for c in classes}

    rows_selected = [r for r in rows if r["fold"] in folds_to_eval]
    print(f"[esc50] decoding {len(rows_selected)} audios (folds={folds_to_eval}) "
          f"at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows_selected:
        try:
            wav = preprocess_audio(r["path"], target_sr, max_audio_samples)
        except Exception as e:
            print(f"[esc50] load fail {r['filename']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    by_fold_total = defaultdict(int)
    by_fold_correct = defaultdict(int)
    by_fold_parsed = defaultdict(int)

    t0 = time.time()
    with open(pred_path, "w") as fp:
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache=use_cache)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyps = []
                for one in batch:
                    hyps.extend(run_batch(model, tokenizer, cfg, [one], max_new_tokens, use_cache=use_cache))

            for r, hyp in zip(batch, hyps):
                pred = parse_single_label(hyp, classes_norm)
                correct = (pred is not None) and (pred == r["label"])
                f = r["fold"]
                by_fold_total[f] += 1
                if pred is not None:
                    by_fold_parsed[f] += 1
                if correct:
                    by_fold_correct[f] += 1
                fp.write(json.dumps({
                    "filename": r["filename"],
                    "fold": f,
                    "label": r["label"],
                    "pred": pred,
                    "raw_output": hyp,
                    "correct": correct,
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                overall_acc = sum(by_fold_correct.values()) / max(done, 1)
                print(f"[esc50] {ckpt_path.name} {done}/{len(prepared)}  "
                      f"acc={overall_acc:.3f}  {rate:.2f} sps  eta {eta/60:.1f}m",
                      flush=True)

    per_fold_acc = {
        str(f): {
            "n": by_fold_total[f],
            "correct": by_fold_correct[f],
            "parsed": by_fold_parsed[f],
            "acc": by_fold_correct[f] / max(by_fold_total[f], 1),
        } for f in sorted(by_fold_total)
    }
    total = sum(v["n"] for v in per_fold_acc.values())
    correct = sum(v["correct"] for v in per_fold_acc.values())
    mean_fold_acc = (sum(v["acc"] for v in per_fold_acc.values()) / len(per_fold_acc)
                     if per_fold_acc else 0.0)

    summary = {
        "checkpoint": str(ckpt_path),
        "folds_evaluated": sorted(by_fold_total.keys()),
        "n_total": total,
        "n_correct": correct,
        "accuracy_pooled": correct / max(total, 1),
        "accuracy_mean_per_fold": mean_fold_acc,
        "per_fold": per_fold_acc,
        "n_classes": len(classes),
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_STEM,
        "use_cache": use_cache,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[esc50] {ckpt_path.name} pooled={summary['accuracy_pooled']:.4f} "
          f"mean-per-fold={mean_fold_acc:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--fold", type=int, default=None,
                   help="Eval only this fold (1-5). Default: see --all.")
    p.add_argument("--all", action="store_true",
                   help="Eval all 5 folds.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable KV/conv cache during generation (shim-free ground truth).")
    p.add_argument("--include-partial", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fold is None and not args.all:
        raise SystemExit("Specify either --fold N or --all")
    folds = list(range(1, 6)) if args.all else [args.fold]

    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    steps_filter = None
    if args.ckpts:
        steps_filter = {int(s) for s in args.ckpts.split(",") if s.strip()}
    ckpts = find_checkpoints(ckpt_root, steps_filter,
                             min_age_sec=0 if args.include_partial else 60)
    if not ckpts:
        raise SystemExit(f"[esc50] no checkpoints under {ckpt_root}")

    rows, classes = load_meta()
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[esc50] classes={len(classes)} rows={len(rows)} folds={folds}", flush=True)

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, rows, classes,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                folds_to_eval=folds,
                use_cache=not args.no_cache,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'ckpt':<24s}  {'acc (pooled)':>13s}  {'mean-per-fold':>14s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        print(f"{name:<24s}  {s['accuracy_pooled']:>13.4f}  {s['accuracy_mean_per_fold']:>14.4f}")


if __name__ == "__main__":
    main()
