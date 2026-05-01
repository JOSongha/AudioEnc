"""LISTEN-official protocol eval for Qwen3.5AE Stage-2 checkpoints.

Faithful port of the LISTEN benchmark reference harness
(https://github.com/DeliJingyiC/LISTEN, scripts/test_your_model.py +
scripts/evaluation_utils.py). Reproduces bit-for-bit:

    - Prompt format:
        {question}

        A. {choice_0}
        B. {choice_1}
        ...

        Respond with only the letter (A, B, C, etc.):
    - random.seed(42) + per-sample random.shuffle of choices
    - Type 3 experiments extend the choice set with 8 canonical emotions
    - Letter parse: first A..H in the response (uppercase)
    - Metrics: sklearn accuracy_score / balanced_accuracy_score /
      f1_score(macro) / f1_score(micro), plus emotion-label normalization
      (happy->happiness etc.) and a marginal-distribution chance baseline.
    - Per-experiment breakdown across {1_audio, 1_text, 1_audio_and_text,
      2A, 2B, 2C, 3A, 3B, 3C}. Type 4 (paralinguistic) is not present in the
      local LISTEN-test parquet and is reported as unavailable.

Data source: local parquet at
/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet (2 635 rows,
the LISTEN-test split held out against our Stage-2 training pool). The
reference harness loads from HF `VibeCheck1/LISTEN_full` train split — our
parquet is a subset of that, excluding type 4.

Usage:
    python -m evaluation.stage2.eval_listen_official \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_listen_official \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 2000 --experiments 2B 3B --batch-size 4
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.stage2._loader import (  # noqa: E402
    audio_sample_rate,
    build_prompt_ids,
    default_max_audio_samples,
    t_audio_for,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
)

DEFAULT_PARQUET = "/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet"
MAX_NEW_TOKENS = 3  # LISTEN reference: letter-only, so 3 tokens is plenty

# Type-3 choice extension, mirrors test_your_model.py:183-184
EMO8 = ["neutral", "sadness", "surprise", "happiness",
        "fear", "anger", "excitement", "frustration"]

# Emotion label normalization, from evaluation_utils.normalize_emotion_labels
EMO_MAP = {
    "anger": "anger",   "angry": "anger",
    "happiness": "happiness", "happy": "happiness",
    "sad": "sadness",   "sadness": "sadness",
    "surprise": "surprise", "pleasant_surprise": "surprise",
    "calm": "calm", "disgust": "disgust", "excitement": "excitement",
    "fear": "fear", "neutral": "neutral", "frustration": "frustration",
}

# Experiment -> (input_mode, parquet filter experiment_type)
# 1_* map to type "1"; 2C/3C reuse 2B/3B audio (add transcription in prompt).
# Type 4 (paralinguistic) was originally absent from the local LISTEN-test parquet;
# 2026-04-30: a combined parquet at /mnt/tmp/listen_analysis/data/test_with_type4.parquet
# adds the 975 type-4 rows from train shard 2. Caveat: type-4 was in the LISTEN-train
# pool used for Stage-2 supervision, so eval on type 4 is upper-bound (training contamination).
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "1_text":            ("text",           "1"),
    "1_audio":           ("audio",          "1"),
    "1_audio_and_text":  ("audio_and_text", "1"),
    "2A":                ("text",           "2A"),
    "2B":                ("audio",          "2B"),
    "2C":                ("audio_and_text", "2B"),
    "3A":                ("text",           "3A"),
    "3B":                ("audio",          "3B"),
    "3C":                ("audio_and_text", "3B"),
    "4":                 ("audio",          "4"),
}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_listen_all(parquet_path: str) -> list[dict]:
    cols = ["id", "question", "answer", "choices", "dataset_source",
            "experiment_type", "transcription", "audio",
            "explicit_emotion"]
    t = pq.read_table(parquet_path, columns=cols)
    rows = []
    for i in range(t.num_rows):
        a = t.column("audio")[i].as_py()
        rows.append({
            "id": t.column("id")[i].as_py(),
            "question": t.column("question")[i].as_py(),
            "answer": t.column("answer")[i].as_py(),
            "choices": t.column("choices")[i].as_py(),
            "source": t.column("dataset_source")[i].as_py(),
            "experiment_type": t.column("experiment_type")[i].as_py(),
            "transcription": t.column("transcription")[i].as_py(),
            "explicit_emotion": t.column("explicit_emotion")[i].as_py(),
            "audio_bytes": a["bytes"] if a else None,
        })
    return rows


def filter_by_exp(rows: list[dict], exp_type_filter: str) -> list[dict]:
    return [r for r in rows if r["experiment_type"] == exp_type_filter]


def decode_audio(wav_bytes: bytes, target_sr: int, max_samples: int) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if wav.shape[-1] > max_samples:
        wav = wav[:max_samples]
    return wav


# ---------------------------------------------------------------------------
# LISTEN choice randomization (reference: test_your_model.py:171-196)
# ---------------------------------------------------------------------------


def randomize_choices(row: dict, exp: str) -> tuple[list[str], str] | None:
    """Return (shuffled_choices, expected_letter) or None if answer not in choices.

    Reproduces the reference's `randomize_choices` exactly:
    - Type 3*: union choices with EMO8
    - For 3A, gold answer is `explicit_emotion`; otherwise it's `answer`
    - random.shuffle is applied to indices of the (optionally extended) list
    """
    base_type = EXPERIMENTS[exp][1]
    choices = list(row["choices"])
    answer = row["explicit_emotion"] if base_type == "3A" else row["answer"]
    if not answer or not str(answer).strip():
        return None

    if base_type in ("3A", "3B"):
        choices = list(set(choices + EMO8))

    if answer not in choices:
        return None

    orig_pos = choices.index(answer)
    indices = list(range(len(choices)))
    random.shuffle(indices)
    new_choices = [choices[i] for i in indices]
    new_pos = indices.index(orig_pos)
    return new_choices, chr(ord("A") + new_pos)


# ---------------------------------------------------------------------------
# LISTEN prompt (reference: test_your_model.py:238-239)
# ---------------------------------------------------------------------------


def listen_prompt(question: str, choices: list[str]) -> str:
    lines = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
    return f"{question}\n\n" + "\n".join(lines) + \
           "\n\nRespond with only the letter (A, B, C, etc.):"


def build_user_suffix(row: dict, shuffled_choices: list[str], input_mode: str) -> str:
    p = listen_prompt(row["question"], shuffled_choices)
    if input_mode in ("text", "audio_and_text"):
        tx = row.get("transcription") or ""
        return f"Transcription: {tx}\n\n{p}"
    return p


# ---------------------------------------------------------------------------
# letter parse — first A-H in response (ref: test_your_model.py:261-265)
# ---------------------------------------------------------------------------


def parse_letter(text: str) -> str | None:
    for ch in text.upper():
        if ch in "ABCDEFGH":
            return ch
    return None


# ---------------------------------------------------------------------------
# metrics (ref: evaluation_utils.py)
# ---------------------------------------------------------------------------


def _norm_lbl(s: str) -> str:
    t = str(s).strip().lower()
    return EMO_MAP.get(t, t)


def compute_metrics(results: list[dict]) -> dict:
    """Reproduces evaluation_utils.calculate_comprehensive_metrics."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    y_true, y_pred = [], []
    for r in results:
        choices = r.get("randomized_choices") or []
        exp = r.get("expected_letter")
        pred = r.get("predicted_letter")
        if not exp or not pred:
            continue
        ie = ord(exp.upper()) - ord("A")
        ip = ord(pred.upper()) - ord("A")
        if not (0 <= ie < len(choices) and 0 <= ip < len(choices)):
            continue
        y_true.append(_norm_lbl(choices[ie]))
        y_pred.append(_norm_lbl(choices[ip]))

    if not y_true:
        return {"weighted_accuracy": 0.0, "uar": 0.0, "macro_f1": 0.0, "micro_f1": 0.0,
                "per_class_accuracy": {}, "chance_baseline":
                {"expected_accuracy": 0.0, "expected_macro_f1": 0.0, "expected_micro_f1": 0.0},
                "n_scored": 0}

    wa = accuracy_score(y_true, y_pred)
    uar = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mif1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    per_class: dict[str, float] = {}
    for lbl in sorted(set(y_true)):
        pairs = [(t, p) for t, p in zip(y_true, y_pred) if t == lbl]
        per_class[lbl] = sum(1 for t, p in pairs if t == p) / len(pairs) if pairs else 0.0

    chance = _chance_baseline(y_true, y_pred)
    return {
        "weighted_accuracy": float(wa),
        "uar": float(uar),
        "macro_f1": float(mf1),
        "micro_f1": float(mif1),
        "per_class_accuracy": per_class,
        "chance_baseline": chance,
        "n_scored": len(y_true),
    }


def _chance_baseline(y_true: list[str], y_pred: list[str]) -> dict:
    """Reproduces evaluation_utils.calculate_chance_baseline + compute_expected_metrics."""
    pred_c = Counter(y_pred)
    true_c = Counter(y_true)
    if not pred_c or not true_c:
        return {"expected_accuracy": 0.0, "expected_macro_f1": 0.0, "expected_micro_f1": 0.0}
    labels = sorted(set(pred_c) | set(true_c))
    Tp = sum(pred_c.values())
    Tt = sum(true_c.values())
    pd_arr = np.array([pred_c.get(l, 0) / Tp for l in labels])
    td_arr = np.array([true_c.get(l, 0) / Tt for l in labels])
    acc = float(np.sum(td_arr * pd_arr))
    f1s = []
    for k in range(len(labels)):
        tp = td_arr[k] * pd_arr[k]
        fp = (1 - td_arr[k]) * pd_arr[k]
        fn = td_arr[k] * (1 - pd_arr[k])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    macro = float(np.mean(f1s))
    # Micro F1 with the reference's formulation
    tp_m = acc
    denom = tp_m + (1 - acc)
    mp = tp_m / denom if denom > 0 else 0.0
    mr = mp  # same as precision by their formula
    micro = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    return {
        "expected_accuracy": acc,
        "expected_macro_f1": macro,
        "expected_micro_f1": float(micro),
    }


# ---------------------------------------------------------------------------
# experiment runner
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_experiment(
    model, tokenizer, cfg,
    rows: list[dict],
    exp: str,
    batch_size: int,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool,
    out_dir: Path,
    max_samples: int | None,
    ckpt_path: Path,
) -> dict:
    input_mode, exp_type_filter = EXPERIMENTS[exp]
    subset = filter_by_exp(rows, exp_type_filter)
    if not subset:
        return {
            "checkpoint": str(ckpt_path),
            "experiment": exp,
            "input_mode": input_mode,
            "available": False,
            "reason": f"no rows with experiment_type={exp_type_filter} in parquet",
        }

    # Re-seed per experiment so within-experiment randomization is deterministic
    # but experiments don't share the RNG state (matches ref: random.seed(42)
    # at main() + iteration over experiments in order).
    random.seed(42)

    if max_samples is not None and max_samples < len(subset):
        subset = subset[:max_samples]

    # Pre-decode audio for modes that need it
    target_sr = audio_sample_rate(cfg)
    need_audio = input_mode in ("audio", "audio_and_text")
    prepared: list[dict] = []
    for r in subset:
        if need_audio and r.get("audio_bytes") is None:
            continue
        try:
            wav = decode_audio(r["audio_bytes"], target_sr, max_audio_samples) if need_audio else None
        except Exception as e:
            print(f"[listen-off] {exp} decode fail {r['id']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})

    # Length-sort audio batches for padding efficiency; text-only doesn't sort.
    if need_audio:
        prepared.sort(key=lambda x: x["_wav"].shape[-1])

    pred_path = out_dir / f"predictions_{exp}.jsonl"
    results: list[dict] = []
    audio_pad_id = cfg.audio_pad_token_id

    t0 = time.time()
    with open(pred_path, "w") as fp:
        i = 0
        while i < len(prepared):
            batch = prepared[i : i + batch_size]
            prompts = []
            waveforms: list[torch.Tensor] | None = [] if need_audio else None
            rand_info: list[dict] = []

            for r in batch:
                rnd = randomize_choices(r, exp)
                if rnd is None:
                    continue
                shuf_choices, expected_letter = rnd
                suffix = build_user_suffix(r, shuf_choices, input_mode)
                t_audio = t_audio_for(cfg, r["_wav"].shape[-1]) if need_audio else 0
                prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, suffix))
                if need_audio:
                    waveforms.append(r["_wav"])
                rand_info.append({
                    "row": r,
                    "shuf_choices": shuf_choices,
                    "expected": expected_letter,
                })

            if not prompts:
                i += batch_size
                continue

            try:
                raw_outs = generate_greedy(
                    model, tokenizer, cfg, prompts, waveforms,
                    max_new_tokens=max_new_tokens, use_cache=use_cache,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raw_outs = []
                for pid, p in enumerate(prompts):
                    w1 = [waveforms[pid]] if waveforms is not None else None
                    raw_outs.extend(generate_greedy(
                        model, tokenizer, cfg, [p], w1,
                        max_new_tokens=max_new_tokens, use_cache=use_cache,
                    ))

            for info, raw in zip(rand_info, raw_outs):
                r = info["row"]
                pred = parse_letter(raw)
                correct = (pred is not None and pred == info["expected"])
                rec = {
                    "sample_id": hashlib.md5(r["id"].encode()).hexdigest()[:8],
                    "source": r["source"],
                    "experiment_type": r["experiment_type"],
                    "input_mode": input_mode,
                    "ground_truth": (r["explicit_emotion"]
                                     if (input_mode == "text" and exp == "3A")
                                     else r["answer"]),
                    "randomized_choices": info["shuf_choices"],
                    "expected_letter": info["expected"],
                    "predicted_letter": pred,
                    "raw_output": raw,
                    "correct": correct,
                }
                results.append(rec)
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            i += batch_size
            if (i // batch_size) % 10 == 0:
                done = min(i, len(prepared))
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                cur_acc = sum(x["correct"] for x in results) / max(len(results), 1)
                print(f"[listen-off] {ckpt_path.name}/{exp} {done}/{len(prepared)}  "
                      f"acc={cur_acc:.3f}  {rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    metrics = compute_metrics(results)
    summary = {
        "checkpoint": str(ckpt_path),
        "experiment": exp,
        "input_mode": input_mode,
        "available": True,
        "n_samples": len(results),
        **metrics,
        "elapsed_sec": time.time() - t0,
        "use_cache": use_cache,
    }
    with open(out_dir / f"summary_{exp}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[listen-off] {ckpt_path.name}/{exp}  WA={metrics['weighted_accuracy']:.4f} "
          f"UAR={metrics['uar']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"(chance={metrics['chance_baseline']['expected_accuracy']:.4f})", flush=True)
    return summary


# ---------------------------------------------------------------------------
# per-ckpt driver
# ---------------------------------------------------------------------------


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    rows: list[dict],
    experiments: list[str],
    batch_size: int,
    out_dir: Path,
    max_samples: int | None,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)

    summaries: dict[str, dict] = {}
    for exp in experiments:
        s = run_experiment(
            model, tokenizer, cfg, rows, exp,
            batch_size=batch_size,
            max_audio_samples=max_audio_samples,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            out_dir=out_dir,
            max_samples=max_samples,
            ckpt_path=ckpt_path,
        )
        summaries[exp] = s

    combined = {"checkpoint": str(ckpt_path), "per_experiment": summaries}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--experiments", nargs="+", default=None,
                   choices=list(EXPERIMENTS.keys()),
                   help="Default: all 9 available experiments")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap rows per experiment (debug)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true")
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
        raise SystemExit(f"[listen-off] no checkpoints under {ckpt_root}")

    rows = load_listen_all(args.parquet)
    exp_counts = Counter(r["experiment_type"] for r in rows)
    print(f"[listen-off] parquet rows={len(rows)}  experiment_type counts={dict(exp_counts)}",
          flush=True)

    experiments = args.experiments or list(EXPERIMENTS.keys())

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        if (out_dir / "summary.json").exists():
            with open(out_dir / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, rows, experiments,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_samples=args.max_samples,
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

    # Tabulate
    print("\n=== SUMMARY ===")
    header = ["ckpt", "exp", "mode", "n", "WA", "UAR", "macroF1", "microF1", "chance"]
    print("  ".join(f"{h:>8s}" if h not in ("exp", "mode") else f"{h:<12s}" for h in header))
    for combo in all_summaries:
        name = Path(combo["checkpoint"]).name
        for exp, s in combo["per_experiment"].items():
            if not s.get("available", False):
                print(f"{name:>8s}  {exp:<12s}  {s.get('input_mode','-'):<12s}  "
                      f"{'skip':>8s}")
                continue
            m = s
            cb = m["chance_baseline"]["expected_accuracy"]
            print(f"{name:>8s}  {exp:<12s}  {m['input_mode']:<12s}  "
                  f"{m['n_samples']:>8d}  {m['weighted_accuracy']:>8.4f}  "
                  f"{m['uar']:>8.4f}  {m['macro_f1']:>8.4f}  "
                  f"{m['micro_f1']:>8.4f}  {cb:>8.4f}")


if __name__ == "__main__":
    main()
