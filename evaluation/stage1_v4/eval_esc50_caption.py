"""ESC-50 5-fold accuracy with caption prompt (matches v4 Stage-1 training).

v4 Stage-1 trains the audio_env_sound modality in caption form (the
``sound_describe_single`` / ``sound_describe_multi`` pools), so the Stage-2
evaluator's ``Classify this sound.`` stem is out-of-distribution. This script
mirrors evaluation.stage2.eval_esc50_acc but with two changes:

  1. Prompt is ``Describe the sound you hear in this audio in one sentence.``
     (TASK_PROMPTS["sound_describe_single"][0]).
  2. The free-form caption is post-processed by substring-matching the 50-class
     ESC-50 vocabulary (word-boundary, longest-match wins, first-match
     tie-break) to recover a single class label.

Everything else (5-fold CV, per-fold + pooled accuracy, audio loading,
model load path, summary schema) is unchanged. Output is drop-in compatible
with the Stage-2 esc50 schema (same keys), so docs/stage2/aggregate_results
already handles it.

Usage:
    python -m evaluation.stage1_v4.eval_esc50_caption \
        --ckpt-root /mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v4 \
        --base-model /mnt/ddn/users/jos/audiollm-trainer/external/models/Qwen3.5AE-4B \
        --out-root  /mnt/tmp/Qwen3.5_dac_vae_v4_Stage1_jos/eval_v4_caption/eval_esc50_caption \
        --ckpts 68000 --all --batch-size 8
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
EVAL_STEM = "Describe the sound you hear in this audio in one sentence."
MAX_NEW_TOKENS = 96


def load_meta() -> tuple[list[dict], list[str]]:
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
                "label": r["category"],
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


def class_patterns(classes: list[str]) -> list[tuple[str, re.Pattern]]:
    """Return [(canonical, compiled_regex)] sorted by surface length desc.

    "vacuum_cleaner" -> r"\bvacuum\s+cleaner(s)?\b"
    "dog"            -> r"\bdog(s)?\b"
    Longer/multi-word patterns are tried first so "vacuum cleaner" wins
    against the standalone "vacuum" cluster (none in ESC-50 but pattern holds).
    """
    pats = []
    for c in classes:
        words = c.replace("_", " ").lower().split()
        body = r"\s+".join(re.escape(w) for w in words)
        pats.append((c, re.compile(rf"\b{body}s?\b", re.IGNORECASE), len(c)))
    pats.sort(key=lambda x: -x[2])
    return [(c, p) for c, p, _ in pats]


def parse_class_from_caption(text: str, patterns: list[tuple[str, re.Pattern]]) -> str | None:
    """First (longest) class whose pattern hits the caption. Caption is
    truncated at the first newline / period so trailing rationale doesn't
    polish the match."""
    head = text.split("\n")[0]
    head = head.split(". ")[0]
    for canon, pat in patterns:
        if pat.search(head):
            return canon
    return None


@torch.inference_mode()
def run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache=True):
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
    max_audio_samples: int | None,
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
    patterns = class_patterns(classes)

    rows_selected = [r for r in rows if r["fold"] in folds_to_eval]
    print(f"[esc50/cap] decoding {len(rows_selected)} audios (folds={folds_to_eval}) "
          f"at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows_selected:
        try:
            wav = preprocess_audio(r["path"], target_sr, max_audio_samples)
        except Exception as e:
            print(f"[esc50/cap] load fail {r['filename']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    by_fold_total = defaultdict(int)
    by_fold_correct = defaultdict(int)
    by_fold_parsed = defaultdict(int)

    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
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
                pred = parse_class_from_caption(hyp, patterns)
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
                print(f"[esc50/cap] {ckpt_path.name} {done}/{len(prepared)}  "
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
        "score_mode": "caption_substring",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[esc50/cap] {ckpt_path.name} pooled={summary['accuracy_pooled']:.4f} "
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
    p.add_argument("--all", action="store_true", help="Eval all 5 folds.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-audio-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all and args.fold is not None:
        raise SystemExit("Specify either --fold N or --all, not both.")
    if not args.all and args.fold is None:
        raise SystemExit("Specify either --fold N or --all")
    folds = [args.fold] if args.fold is not None else [1, 2, 3, 4, 5]

    ckpt_paths = find_checkpoints(Path(args.ckpt_root), args.ckpts)
    if not ckpt_paths:
        raise SystemExit(f"[esc50/cap] no checkpoints under {args.ckpt_root}")

    rows, classes = load_meta()
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[esc50/cap] {len(rows)} ESC-50 clips, {len(classes)} classes",
          flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for ckpt_path in ckpt_paths:
        out_dir = out_root / ckpt_path.name
        if (out_dir / "summary.json").exists():
            print(f"[esc50/cap] skip {ckpt_path.name} (summary exists)", flush=True)
            continue
        eval_checkpoint(
            ckpt_path=ckpt_path,
            base_model=args.base_model,
            rows=rows,
            classes=classes,
            batch_size=args.batch_size,
            out_dir=out_dir,
            max_audio_samples=args.max_audio_samples,
            max_new_tokens=args.max_new_tokens,
            folds_to_eval=folds,
            use_cache=not args.no_cache,
        )


if __name__ == "__main__":
    main()
