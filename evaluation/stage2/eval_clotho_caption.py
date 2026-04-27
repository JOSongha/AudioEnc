"""Clotho evaluation-split captioning eval for Qwen3.5AE Stage-2 checkpoints.

Prompt mirrors training: the canonical sound-caption stem from TASK_PROMPTS.
Generates one caption per clip via greedy decode; scores against the 5 Clotho
reference captions with corpus BLEU-1..4 (implemented inline) and — if
pycocoevalcap is importable — CIDEr / METEOR / SPICE. Skipped metrics are
reported as None in summary.json.

Usage:
    python -m evaluation.stage2.eval_clotho_caption \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_clotho \
        --base-model /mnt/tmp/s2_init_42k \
        --split evaluation \
        --ckpts 2000 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

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

CLOTHO_ROOT = Path("/mnt/tmp/datasets/env_sound/Clotho")
# Canonical stem — TASK_PROMPTS["sound_caption"][0]. Pinned here to avoid
# cross-module imports.
EVAL_CAPTION_STEM = "Describe what you hear in the audio."
MAX_NEW_TOKENS = 96


# ---------------------------------------------------------------------------
# BLEU-N corpus score (simple inline, matches NLTK/COCO corpus-BLEU behavior)
# ---------------------------------------------------------------------------


_TOK_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(s: str) -> list[str]:
    return [w.lower() for w in _TOK_RE.findall(s)]


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(list_of_refs: list[list[list[str]]], hyps: list[list[str]],
                max_n: int = 4) -> tuple[float, list[float]]:
    """Sacre-style corpus BLEU. Returns (BLEU-N, per-order precisions).

    list_of_refs[i] is a list of reference token-lists for hyp i.
    """
    assert len(list_of_refs) == len(hyps)
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for refs, hyp in zip(list_of_refs, hyps):
        hyp_len += len(hyp)
        # Closest ref length for BP.
        ref_len += min(
            (abs(len(r) - len(hyp)), len(r)) for r in refs
        )[1] if refs else len(hyp)

        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            max_ref_ng: Counter = Counter()
            for r in refs:
                r_ng = _ngrams(r, n)
                for k, v in r_ng.items():
                    if v > max_ref_ng[k]:
                        max_ref_ng[k] = v
            for ng, c in hyp_ng.items():
                clipped[n - 1] += min(c, max_ref_ng[ng])
            totals[n - 1] += sum(hyp_ng.values())

    precisions = []
    for n in range(max_n):
        if totals[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped[n] / totals[n])

    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return bp * geo, precisions


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_clotho_split(split: str) -> list[dict]:
    """Return list of {file_path, captions[5]}."""
    csv_path = CLOTHO_ROOT / f"captions_{split}.csv"
    audio_dir = CLOTHO_ROOT / split
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = r["file_name"]
            ap = audio_dir / fn
            if not ap.exists():
                continue
            caps = [r[f"caption_{i}"] for i in range(1, 6) if r.get(f"caption_{i}")]
            rows.append({"file": fn, "path": str(ap), "captions": caps})
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
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, EVAL_CAPTION_STEM))
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
    )


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    rows: list[dict],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int,
    max_new_tokens: int,
    try_pycoco: bool,
    use_cache: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)

    # Pre-decode audio and sort for batch stability.
    print(f"[clotho] decoding {len(rows)} audios...", flush=True)
    prepared = []
    for r in rows:
        try:
            wav = preprocess_audio(r["path"], max_audio_samples)
        except Exception as e:
            print(f"[clotho] load fail {r['file']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    all_preds: list[str] = []
    all_refs: list[list[list[str]]] = []  # per-sample list of tokenized refs

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
                all_preds.append(hyp)
                all_refs.append([tokenize(c) for c in r["captions"]])
                fp.write(json.dumps({
                    "file": r["file"],
                    "hyp": hyp,
                    "refs": r["captions"],
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 10 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                print(f"[clotho] {ckpt_path.name} {done}/{len(prepared)}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    # Corpus BLEU (inline).
    hyp_toks = [tokenize(h) for h in all_preds]
    bleu4, precisions = corpus_bleu(all_refs, hyp_toks, max_n=4)
    bleu1 = precisions[0] if precisions else 0.0

    # Optional COCO-style CIDEr/METEOR/ROUGE/SPICE via pycocoevalcap.
    # Each metric attempted separately so Java-only ones (METEOR, SPICE) can
    # fail without dropping CIDEr/ROUGE.
    coco_scores: dict[str, float | None] = {"CIDEr": None, "METEOR": None,
                                             "ROUGE_L": None, "SPICE": None}
    if try_pycoco:
        gts = {str(i): r["captions"] for i, r in enumerate(prepared)}
        res = {str(i): [all_preds[i]] for i in range(len(all_preds))}

        try:
            from pycocoevalcap.cider.cider import Cider
            coco_scores["CIDEr"] = float(Cider().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[clotho] CIDEr unavailable: {e}", flush=True)

        try:
            from pycocoevalcap.rouge.rouge import Rouge
            coco_scores["ROUGE_L"] = float(Rouge().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[clotho] ROUGE_L unavailable: {e}", flush=True)

        try:
            from pycocoevalcap.meteor.meteor import Meteor
            coco_scores["METEOR"] = float(Meteor().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[clotho] METEOR unavailable (needs Java): {e}", flush=True)

        try:
            from pycocoevalcap.spice.spice import Spice
            coco_scores["SPICE"] = float(Spice().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[clotho] SPICE unavailable (needs Java + CoreNLP): {e}", flush=True)

    summary = {
        "checkpoint": str(ckpt_path),
        "n": len(prepared),
        "bleu1": bleu1,
        "bleu4": bleu4,
        "precisions_1to4": precisions,
        **coco_scores,
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_CAPTION_STEM,
        "max_audio_samples": max_audio_samples,
        "use_cache": use_cache,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[clotho] {ckpt_path.name} BLEU-1={bleu1:.4f} BLEU-4={bleu4:.4f} "
          f"CIDEr={coco_scores['CIDEr']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--split", default="evaluation", choices=["evaluation", "validation"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=1_600_000)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-pycoco", action="store_true",
                   help="Skip the CIDEr/METEOR/ROUGE block even if pycocoevalcap is installed.")
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
        raise SystemExit(f"[clotho] no checkpoints under {ckpt_root}")

    rows = load_clotho_split(args.split)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[clotho] split={args.split} rows={len(rows)}", flush=True)

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
                p, args.base_model, rows,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                try_pycoco=not args.no_pycoco,
                use_cache=not args.no_cache,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'ckpt':<24s}  {'BLEU-1':>7s}  {'BLEU-4':>7s}  {'CIDEr':>7s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        c = s.get("CIDEr")
        c_str = f"{c:.4f}" if isinstance(c, (int, float)) else "   -   "
        print(f"{name:<24s}  {s['bleu1']:>7.4f}  {s['bleu4']:>7.4f}  {c_str:>7s}")


if __name__ == "__main__":
    main()
