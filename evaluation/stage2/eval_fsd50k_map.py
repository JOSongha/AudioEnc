"""FSD50K eval-split sound-event multi-label eval for Qwen3.5AE Stage-2.

IMPORTANT CAVEAT on metric choice:
    Published FSD50K mAP requires per-label CONFIDENCE scores to sweep thresholds.
    Our generative setup only produces a hard label list (commas-separated). You
    can recover a presence-only binary prediction vector, so we score with
    label-wise F1 (micro + macro) and Jaccard similarity. If you need true mAP
    you must replace greedy sampling with a token-level logit readout over the
    200 vocabulary terms — out of scope here.

Prompt: TASK_PROMPTS["sound_classify_multi"][0] — matches training.
Target parse: split generated string by comma, normalize (lowercase, replace
    spaces with '_'), intersect with the FSD50K 200-class vocabulary.

Usage:
    python -m evaluation.stage2.eval_fsd50k_map \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_fsd50k \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 2000 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.stage2._loader import (  # noqa: E402
    HOP_LENGTH,
    SAMPLE_RATE,
    build_prompt_ids,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
)

FSD50K_ROOT = Path("/mnt/tmp/datasets/env_sound/FSD50K")
EVAL_CSV = FSD50K_ROOT / "FSD50K.ground_truth/eval.csv"
VOCAB_CSV = FSD50K_ROOT / "FSD50K.ground_truth/vocabulary.csv"
EVAL_AUDIO_DIR = FSD50K_ROOT / "FSD50K.eval_audio"

EVAL_STEM = "List the sound events in this audio, separated by commas."
MAX_NEW_TOKENS = 96


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_vocab() -> tuple[list[str], dict[str, int]]:
    """Return (label_list, name -> index)."""
    labels = []
    with open(VOCAB_CSV) as f:
        for row in csv.reader(f):
            # vocabulary.csv has no header: idx,label_name,mid
            if len(row) < 2:
                continue
            labels.append(row[1])
    name_to_idx = {n.lower(): i for i, n in enumerate(labels)}
    return labels, name_to_idx


def load_eval(max_samples: int | None) -> list[dict]:
    rows = []
    with open(EVAL_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r["fname"]
            ap = EVAL_AUDIO_DIR / f"{fname}.wav"
            if not ap.exists():
                continue
            labels = [l for l in r["labels"].split(",") if l]
            rows.append({"fname": fname, "path": str(ap), "labels": labels})
    if max_samples:
        rows = rows[:max_samples]
    return rows


def preprocess_audio(path: str, max_samples: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if wav.shape[-1] > max_samples:
        wav = wav[:max_samples]
    return wav


# ---------------------------------------------------------------------------
# label normalization / parsing
# ---------------------------------------------------------------------------


_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_label(s: str) -> str:
    return _NORM_RE.sub("_", s.strip().lower()).strip("_")


def parse_predicted_labels(text: str, vocab_norm: dict[str, int]) -> list[int]:
    """Split generated text by commas, normalize each, keep known vocab indices."""
    # Stop at newline / sentence break to avoid parsing rationale text if any.
    head = text.split("\n")[0]
    pieces = [p for p in head.split(",") if p.strip()]
    idxs: set[int] = set()
    for p in pieces:
        key = norm_label(p)
        if not key:
            continue
        if key in vocab_norm:
            idxs.add(vocab_norm[key])
        else:
            # Try each space-separated sub-token ("dog bark" often normalizes to "dog_bark")
            # already handled by norm_label. Also try last word alone.
            tail = key.split("_")[-1]
            if tail in vocab_norm:
                idxs.add(vocab_norm[tail])
    return sorted(idxs)


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
        t_audio = max(1, wav.shape[-1] // HOP_LENGTH)
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
    vocab: list[str],
    vocab_norm: dict[str, int],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)

    print(f"[fsd50k] decoding {len(rows)} audios...", flush=True)
    prepared = []
    for r in rows:
        try:
            wav = preprocess_audio(r["path"], max_audio_samples)
        except Exception as e:
            print(f"[fsd50k] load fail {r['fname']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    n_labels = len(vocab)
    y_true = np.zeros((len(prepared), n_labels), dtype=np.uint8)
    y_pred = np.zeros((len(prepared), n_labels), dtype=np.uint8)

    t0 = time.time()
    with open(pred_path, "w") as fp:
        row_i = 0
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
                pred_idxs = parse_predicted_labels(hyp, vocab_norm)
                true_idxs = [vocab_norm[norm_label(l)] for l in r["labels"]
                             if norm_label(l) in vocab_norm]
                for j in true_idxs:
                    y_true[row_i, j] = 1
                for j in pred_idxs:
                    y_pred[row_i, j] = 1
                fp.write(json.dumps({
                    "fname": r["fname"],
                    "labels_true": r["labels"],
                    "labels_pred": [vocab[j] for j in pred_idxs],
                    "raw_output": hyp,
                }, ensure_ascii=False) + "\n")
                row_i += 1

            done = i + len(batch)
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                print(f"[fsd50k] {ckpt_path.name} {done}/{len(prepared)}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    # F1 + Jaccard. Hand-compute to avoid sklearn dep on specific version.
    tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
    fp_c = ((1 - y_true) & y_pred).sum(axis=0).astype(np.float64)
    fn_c = (y_true & (1 - y_pred)).sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp_c > 0, tp / (tp + fp_c), 0.0)
        rec = np.where(tp + fn_c > 0, tp / (tp + fn_c), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    # Micro (pool all samples' labels)
    tp_mi = tp.sum()
    fp_mi = fp_c.sum()
    fn_mi = fn_c.sum()
    prec_mi = tp_mi / (tp_mi + fp_mi) if (tp_mi + fp_mi) else 0.0
    rec_mi = tp_mi / (tp_mi + fn_mi) if (tp_mi + fn_mi) else 0.0
    f1_micro = 2 * prec_mi * rec_mi / (prec_mi + rec_mi) if (prec_mi + rec_mi) else 0.0

    # Macro (average over labels with support > 0).
    has_support = (tp + fn_c) > 0
    f1_macro = f1[has_support].mean() if has_support.any() else 0.0

    # Sample-level Jaccard.
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    jac = np.where(union > 0, inter / union, 1.0)
    jaccard = float(jac.mean())

    # Average number of true/predicted labels per sample.
    mean_true = float(y_true.sum(axis=1).mean())
    mean_pred = float(y_pred.sum(axis=1).mean())

    summary = {
        "checkpoint": str(ckpt_path),
        "n": int(row_i),
        "n_labels": n_labels,
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(prec_mi),
        "recall_micro": float(rec_mi),
        "jaccard_mean": jaccard,
        "avg_labels_true": mean_true,
        "avg_labels_pred": mean_pred,
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_STEM,
        "note": "Generative pipeline cannot produce per-label confidences; "
                "mAP unavailable. F1/Jaccard reported instead.",
        "use_cache": use_cache,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[fsd50k] {ckpt_path.name} F1-micro={f1_micro:.4f} "
          f"F1-macro={f1_macro:.4f} Jaccard={jaccard:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=1_600_000)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable KV/conv cache during generation (shim-free ground truth).")
    p.add_argument("--include-partial", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    steps_filter = None
    if args.ckpts:
        steps_filter = {int(s) for s in args.ckpts.split(",") if s.strip()}
    ckpts = find_checkpoints(ckpt_root, steps_filter,
                             min_age_sec=0 if args.include_partial else 60)
    if not ckpts:
        raise SystemExit(f"[fsd50k] no checkpoints under {ckpt_root}")

    vocab, vocab_norm = load_vocab()
    rows = load_eval(args.max_samples)
    print(f"[fsd50k] vocab={len(vocab)} rows={len(rows)}", flush=True)

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
                p, args.base_model, rows, vocab, vocab_norm,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_cache,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'ckpt':<24s}  {'F1-mi':>6s}  {'F1-ma':>6s}  {'Jacc':>6s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        print(f"{name:<24s}  {s['f1_micro']:>6.4f}  {s['f1_macro']:>6.4f}  "
              f"{s['jaccard_mean']:>6.4f}")


if __name__ == "__main__":
    main()
