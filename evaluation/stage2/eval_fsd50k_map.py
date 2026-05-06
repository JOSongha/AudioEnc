"""FSD50K eval-split sound-event multi-label eval for Qwen3.5AE Stage-2.

Two scoring modes, selectable via `--score-mode`:

1. `greedy` (default): model generates a comma-separated label list; we
   parse, normalize, intersect with the 200-class vocabulary, and report
   F1-micro / F1-macro / Jaccard on presence-only binary vectors. Fast
   (1 forward per sample) but produces no per-label probability, so mAP
   is not recoverable.
2. `sequence`: teacher-forced scoring of every label. For each sample we
   compute sum(log P(label_tok[j] | prompt + audio + label_tok[:j])) for
   each of the 200 label strings, length-normalized. These log-probs rank
   samples per label and feed sklearn's `average_precision_score` to
   produce the classical FSD50K mAP. Slower (~200 / `label_batch_size`
   forwards per sample) but directly leaderboard-comparable.

Both modes use the same prompt format (mirrors training's
`TASK_PROMPTS["sound_classify_multi"][0]`) and the same FSD50K vocabulary
normalizer, so their label-name matching is consistent.

Usage:
    # F1/Jaccard (fast):
    python -m evaluation.stage2.eval_fsd50k_map \
        --ckpt-root .../results/... --out-root .../eval_fsd50k \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 2000 --batch-size 4 --score-mode greedy

    # mAP (slow, leaderboard-comparable):
    python -m evaluation.stage2.eval_fsd50k_map \
        --ckpt-root .../results/... --out-root .../eval_fsd50k_map \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 2000 --score-mode sequence --label-batch-size 50
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
    audio_sample_rate,
    build_prompt_ids,
    default_max_audio_samples,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
    t_audio_for,
    score_labels_teacher_forced,
)

FSD50K_ROOT = Path("/mnt/tmp/datasets/env_sound/FSD50K")
EVAL_CSV = FSD50K_ROOT / "FSD50K.ground_truth/eval.csv"
VOCAB_CSV = FSD50K_ROOT / "FSD50K.ground_truth/vocabulary.csv"
# Override with FSD50K_EVAL_AUDIO_DIR env var (e.g. /dev/shm/FSD50K_eval_audio)
# to bypass disk-seek contention when running many parallel evals.
import os as _os
EVAL_AUDIO_DIR = Path(_os.environ.get(
    "FSD50K_EVAL_AUDIO_DIR",
    str(FSD50K_ROOT / "FSD50K.eval_audio"),
))

EVAL_STEM = "List the sound events in this audio, separated by commas."
SENTENCE_STEM = "Describe what you hear in this audio. Mention every distinct sound event."
MAX_NEW_TOKENS = 96
SENTENCE_MAX_NEW_TOKENS = 256


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
              max_new_tokens: int, use_cache: bool = True,
              stem: str = EVAL_STEM) -> list[str]:
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in batch:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, stem))
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
    )


def parse_labels_sentence(text: str, vocab: list[str]) -> list[int]:
    """Sentence mode: substring-match each label name in the free-form description.
    Word-boundary aware; tolerates plurals and underscore-vs-space."""
    text_lower = text.lower()
    out = set()
    for idx, name in enumerate(vocab):
        # FSD50K names use underscores e.g. "Computer_keyboard" -- normalize to space
        clean = name.lower().replace("_", " ").replace("-", " ").strip()
        if not clean:
            continue
        words = clean.split()
        pat = r"\b" + r"\s+".join(re.escape(w) for w in words[:-1] + [words[-1]]) + r"s?\b"
        if re.search(pat, text_lower):
            out.add(idx)
    return sorted(out)


def _fsd_label_token_seqs(tokenizer, vocab: list[str]) -> list[list[int]]:
    """Pre-tokenize each vocab label for teacher-forced scoring. The leading
    space makes the first token BPE-match the typical continuation context."""
    out = []
    for v in vocab:
        # Humanize: "Computer_keyboard" -> "Computer keyboard" (training target
        # used the raw vocab strings joined by ", "). Keep underscores to match
        # training target exactly.
        out.append(tokenizer.encode(v, add_special_tokens=False))
    return out


def _run_sequence_scoring(
    model, tokenizer, cfg,
    prepared: list[dict],
    vocab: list[str],
    vocab_norm: dict[str, int],
    out_dir: Path,
    ckpt_path: Path,
    label_batch_size: int,
) -> dict:
    """Score all 200 labels per sample via teacher-forced log-prob, compute mAP."""
    from sklearn.metrics import average_precision_score  # deferred

    n_labels = len(vocab)
    label_token_seqs = _fsd_label_token_seqs(tokenizer, vocab)
    audio_pad_id = cfg.audio_pad_token_id

    # y_true (binary) and y_score (log-prob) matrices.
    N = len(prepared)
    y_true = np.zeros((N, n_labels), dtype=np.uint8)
    y_score = np.full((N, n_labels), -1e9, dtype=np.float32)

    pred_path = out_dir / "predictions_seq.jsonl"
    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        for row_i, r in enumerate(prepared):
            true_idxs = [vocab_norm[norm_label(l)] for l in r["labels"]
                         if norm_label(l) in vocab_norm]
            for j in true_idxs:
                y_true[row_i, j] = 1

            t_audio = t_audio_for(cfg, r["_wav"].shape[-1])
            prompt_ids = build_prompt_ids(tokenizer, audio_pad_id, t_audio, EVAL_STEM)

            try:
                scores = score_labels_teacher_forced(
                    model, cfg, prompt_ids, r["_wav"], label_token_seqs,
                    batch_size=label_batch_size, length_normalize=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                scores = score_labels_teacher_forced(
                    model, cfg, prompt_ids, r["_wav"], label_token_seqs,
                    batch_size=max(1, label_batch_size // 4),
                    length_normalize=True,
                )
            y_score[row_i] = scores.cpu().numpy()

            # Top-5 predictions for qualitative inspection (not used for mAP)
            top5 = np.argsort(-y_score[row_i])[:5].tolist()
            fp.write(json.dumps({
                "fname": r["fname"],
                "labels_true": r["labels"],
                "top5_pred_labels": [vocab[j] for j in top5],
                "top5_scores": [float(y_score[row_i, j]) for j in top5],
            }, ensure_ascii=False) + "\n")

            if row_i % 10 == 0:
                dt = time.time() - t0
                rate = (row_i + 1) / max(dt, 1e-6)
                eta = (N - row_i - 1) / max(rate, 1e-6)
                print(f"[fsd50k-seq] {ckpt_path.name} {row_i+1}/{N}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    # mAP
    # sklearn's average_precision_score with average="macro" averages AP over
    # labels with support. For labels with zero positives, AP is undefined —
    # skip them in macro-average.
    valid_mask = y_true.sum(axis=0) > 0
    y_true_valid = y_true[:, valid_mask]
    y_score_valid = y_score[:, valid_mask]

    # Per-label AP
    per_label_ap = []
    for k in range(y_true_valid.shape[1]):
        yt = y_true_valid[:, k]
        if yt.sum() == 0:
            continue
        ys = y_score_valid[:, k]
        ap = average_precision_score(yt, ys)
        per_label_ap.append(ap)
    mAP_macro = float(np.mean(per_label_ap)) if per_label_ap else 0.0
    mAP_micro = float(average_precision_score(
        y_true_valid.ravel(), y_score_valid.ravel()))

    summary = {
        "checkpoint": str(ckpt_path),
        "n": int(N),
        "n_labels": n_labels,
        "n_labels_with_support": int(valid_mask.sum()),
        "mAP_macro": mAP_macro,
        "mAP_micro": mAP_micro,
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_STEM,
        "score_mode": "sequence",
        "note": "Teacher-forced per-label log-prob scoring (length-normalized); "
                "ranking scores feed sklearn.average_precision_score.",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # Also save raw score matrix (useful for re-analysis)
    np.savez(out_dir / "scores.npz",
             y_true=y_true, y_score=y_score, labels=np.array(vocab))
    print(f"[fsd50k-seq] {ckpt_path.name} mAP-macro={mAP_macro:.4f} "
          f"mAP-micro={mAP_micro:.4f} n_labels_scored={valid_mask.sum()}",
          flush=True)
    return summary


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
    score_mode: str = "greedy",
    label_batch_size: int = 50,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    target_sr = audio_sample_rate(cfg)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)

    # Optional: load pre-decoded waveforms from /dev/shm cache (set
    # FSD50K_DECODED_CACHE env). Speeds up dense parallel sweeps by
    # eliminating per-process redundant resample work. Cache is
    # encoder-specific (DAC 48 kHz vs Whisper 16 kHz) — caller's responsibility.
    cache_path = _os.environ.get("FSD50K_DECODED_CACHE")
    cache: dict[str, torch.Tensor] | None = None
    if cache_path and Path(cache_path).exists():
        print(f"[fsd50k] loading pre-decoded cache <- {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu")
        print(f"[fsd50k] cache: {len(cache)} entries", flush=True)

    print(f"[fsd50k] decoding {len(rows)} audios at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows:
        try:
            if cache is not None:
                wav_fp16 = cache.get(r["fname"])
                if wav_fp16 is None:
                    wav = preprocess_audio(r["path"], target_sr, max_audio_samples)
                else:
                    wav = wav_fp16.to(torch.float32)
            else:
                wav = preprocess_audio(r["path"], target_sr, max_audio_samples)
        except Exception as e:
            print(f"[fsd50k] load fail {r['fname']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    if score_mode == "sequence":
        s = _run_sequence_scoring(
            model, tokenizer, cfg, prepared, vocab, vocab_norm,
            out_dir=out_dir, ckpt_path=ckpt_path,
            label_batch_size=label_batch_size,
        )
        del model
        torch.cuda.empty_cache()
        return s

    # greedy / sentence path
    if score_mode == "sentence":
        active_stem = SENTENCE_STEM
        active_max_new = max_new_tokens or SENTENCE_MAX_NEW_TOKENS
        parse_fn = lambda txt: parse_labels_sentence(txt, vocab)
    else:
        active_stem = EVAL_STEM
        active_max_new = max_new_tokens
        parse_fn = lambda txt: parse_predicted_labels(txt, vocab_norm)

    n_labels = len(vocab)
    y_true = np.zeros((len(prepared), n_labels), dtype=np.uint8)
    y_pred = np.zeros((len(prepared), n_labels), dtype=np.uint8)

    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        row_i = 0
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch, active_max_new, use_cache=use_cache, stem=active_stem)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyps = []
                for one in batch:
                    hyps.extend(run_batch(model, tokenizer, cfg, [one], active_max_new, use_cache=use_cache, stem=active_stem))

            for r, hyp in zip(batch, hyps):
                pred_idxs = parse_fn(hyp)
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
        "stem": active_stem,
        "note": ("score_mode=sentence: free-form description with substring label match" if score_mode == "sentence"
                 else "score_mode=greedy: model emits hard label list; F1/Jaccard reported. For mAP use --score-mode sequence."),
        "score_mode": score_mode,
        "use_cache": use_cache,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
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
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable KV/conv cache during generation (shim-free ground truth).")
    p.add_argument("--score-mode", choices=["greedy", "sequence", "sentence"], default="greedy",
                   help="greedy = parse generated label list (F1/Jaccard); "
                        "sequence = teacher-forced per-label log-prob (mAP).")
    p.add_argument("--label-batch-size", type=int, default=50,
                   help="Label sub-batch for sequence scoring "
                        "(200 labels total; memory/speed tradeoff).")
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
                score_mode=args.score_mode,
                label_batch_size=args.label_batch_size,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    if args.score_mode == "sequence":
        print(f"{'ckpt':<24s}  {'mAP-macro':>10s}  {'mAP-micro':>10s}  {'n_labels':>10s}")
        for s in all_summaries:
            name = Path(s["checkpoint"]).name
            print(f"{name:<24s}  {s.get('mAP_macro', float('nan')):>10.4f}  "
                  f"{s.get('mAP_micro', float('nan')):>10.4f}  "
                  f"{s.get('n_labels_with_support', 0):>10d}")
    else:
        print(f"{'ckpt':<24s}  {'F1-mi':>6s}  {'F1-ma':>6s}  {'Jacc':>6s}")
        for s in all_summaries:
            name = Path(s["checkpoint"]).name
            print(f"{name:<24s}  {s.get('f1_micro', float('nan')):>6.4f}  "
                  f"{s.get('f1_macro', float('nan')):>6.4f}  "
                  f"{s.get('jaccard_mean', float('nan')):>6.4f}")


if __name__ == "__main__":
    main()
