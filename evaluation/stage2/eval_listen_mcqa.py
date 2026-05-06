"""LISTEN-test MCQA eval for Qwen3.5AE Stage 2 checkpoints.

Mirrors the training prompt format from omni_dataset.py for audio_emotion rows:
    <stem>
    Choices: A) <c1> B) <c2> ...
    Answer with the letter.

The model generates `<letter>. <rationale>` (training-time target). We parse the
first A-I letter in the response and map it back to the choice index.

Usage:
    python -m evaluation.stage2.eval_listen_mcqa \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/eval_listen \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 1000,2000 \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
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

DEFAULT_PARQUET = "/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet"
# Canonical eval stem. TASK_PROMPTS["emotion_classify"][0] is the same string —
# pinning here avoids a cross-module import dep.
EVAL_EMOTION_STEM = "What emotion is being expressed in the audio?"

LETTER_RE = re.compile(r"\b([A-I])\b")
MAX_NEW_TOKENS = 96  # letter + short rationale


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_listen_test(parquet_path: str) -> list[dict]:
    cols = [
        "id",
        "question",
        "answer",
        "choices",
        "dataset_source",
        "experiment_type",
        "transcription",
        "audio",
    ]
    table = pq.read_table(parquet_path, columns=cols)
    rows = []
    n = table.num_rows
    for i in range(n):
        audio_struct = table.column("audio")[i].as_py()
        rows.append({
            "id": table.column("id")[i].as_py(),
            "question": table.column("question")[i].as_py(),
            "answer_text": table.column("answer")[i].as_py(),
            "choices": table.column("choices")[i].as_py(),
            "source": table.column("dataset_source")[i].as_py(),
            "experiment_type": table.column("experiment_type")[i].as_py(),
            "transcript": table.column("transcription")[i].as_py(),
            "audio_bytes": audio_struct["bytes"] if audio_struct else None,
        })
    return rows


def preprocess_audio(wav_bytes: bytes, target_sr: int, max_samples: int | None = None) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if max_samples is not None and wav.shape[-1] > max_samples:
        wav = wav[:max_samples]
    return wav


# ---------------------------------------------------------------------------
# prompt + answer parsing
# ---------------------------------------------------------------------------


def render_choices(choices: list[str]) -> tuple[str, list[str]]:
    """Return (space-joined 'A) x B) y ...', list of letters aligned with choices)."""
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    pieces = [f"{ltr}) {c}" for ltr, c in zip(letters, choices)]
    return " ".join(pieces), letters


def build_user_suffix(choices: list[str], use_native_question: bool, native_question: str) -> str:
    choices_str, _ = render_choices(choices)
    stem = native_question if use_native_question else EVAL_EMOTION_STEM
    return f"{stem}\nChoices: {choices_str}\nAnswer with the letter."


def parse_letter(text: str, n_choices: int) -> str | None:
    """First A-I letter (bounded by word boundary) within the first 20 chars."""
    # Prefer a letter that appears right at the start (model trained with
    # "<letter>. rationale" format, so leading letter is most reliable).
    head = text.lstrip()[:20]
    m = LETTER_RE.search(head)
    if m:
        c = m.group(1)
        if ord(c) - ord("A") < n_choices:
            return c
    # Fallback anywhere.
    m = LETTER_RE.search(text)
    if m:
        c = m.group(1)
        if ord(c) - ord("A") < n_choices:
            return c
    return None


def letter_from_text_answer(answer_text: str, choices: list[str]) -> str | None:
    """Map a gold text label to its lettered index. Case-insensitive equality match."""
    at = answer_text.strip().lower()
    for i, c in enumerate(choices):
        if c.strip().lower() == at:
            return chr(ord("A") + i)
    return None


# ---------------------------------------------------------------------------
# evaluation loop (single checkpoint)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_batch(
    model,
    tokenizer,
    cfg,
    rows: list[dict],
    use_native_question: bool,
    max_new_tokens: int,
    use_cache: bool = True,
) -> list[str]:
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in rows:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        user_suffix = build_user_suffix(
            r["choices"], use_native_question, r["question"]
        )
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, user_suffix))
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
    use_native_question: bool,
    max_samples: int | None,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    target_sr = audio_sample_rate(cfg)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)

    # Subsample + decode audio upfront (slow path; keep outside of the inner loop).
    if max_samples is not None and max_samples < len(rows):
        rows = rows[:max_samples]
    print(f"[listen] decoding {len(rows)} audios at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows:
        if r.get("audio_bytes") is None:
            continue
        try:
            wav = preprocess_audio(r["audio_bytes"], target_sr, max_samples=max_audio_samples)
        except Exception as e:
            print(f"[listen] audio decode fail on {r['id']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})

    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    n_correct = 0
    n_parsed = 0
    by_source_total = Counter()
    by_source_correct = Counter()
    by_source_parsed = Counter()
    # For macro-F1 on overall label vocabulary
    label_true: list[str] = []
    label_pred: list[str] = []

    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            try:
                raw_outs = run_batch(
                    model, tokenizer, cfg, batch,
                    use_native_question=use_native_question,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raw_outs = []
                for one in batch:
                    raw_outs.extend(run_batch(
                        model, tokenizer, cfg, [one],
                        use_native_question=use_native_question,
                        max_new_tokens=max_new_tokens,
                        use_cache=use_cache,
                    ))

            for r, gen_text in zip(batch, raw_outs):
                gold_letter = letter_from_text_answer(r["answer_text"], r["choices"])
                pred_letter = parse_letter(gen_text, len(r["choices"]))
                is_correct = (pred_letter is not None
                              and gold_letter is not None
                              and pred_letter == gold_letter)
                src = r["source"]
                by_source_total[src] += 1
                if pred_letter is not None:
                    n_parsed += 1
                    by_source_parsed[src] += 1
                if is_correct:
                    n_correct += 1
                    by_source_correct[src] += 1

                pred_text = (r["choices"][ord(pred_letter) - ord("A")]
                             if pred_letter is not None else None)
                label_true.append(r["answer_text"].strip().lower())
                label_pred.append(
                    pred_text.strip().lower() if pred_text is not None else ""
                )

                fp.write(json.dumps({
                    "id": r["id"],
                    "source": src,
                    "experiment_type": r["experiment_type"],
                    "question": r["question"],
                    "choices": r["choices"],
                    "gold_answer": r["answer_text"],
                    "gold_letter": gold_letter,
                    "pred_letter": pred_letter,
                    "pred_answer": pred_text,
                    "raw_output": gen_text,
                    "correct": is_correct,
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 10 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                acc = n_correct / max(done, 1)
                print(f"[listen] {ckpt_path.name} {done}/{len(prepared)}  "
                      f"acc={acc:.3f} parsed={n_parsed}/{done}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    # Macro-F1 over observed labels
    labels_observed = sorted({l for l in label_true} | {p for p in label_pred if p})
    per_label = {}
    f1_sum = 0.0
    valid_labels = 0
    for lbl in labels_observed:
        tp = sum(1 for t, p in zip(label_true, label_pred) if t == lbl and p == lbl)
        fp_c = sum(1 for t, p in zip(label_true, label_pred) if t != lbl and p == lbl)
        fn_c = sum(1 for t, p in zip(label_true, label_pred) if t == lbl and p != lbl)
        if tp + fp_c + fn_c == 0:
            continue
        prec = tp / (tp + fp_c) if (tp + fp_c) else 0.0
        rec = tp / (tp + fn_c) if (tp + fn_c) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lbl] = {"support": tp + fn_c, "precision": prec, "recall": rec, "f1": f1}
        if tp + fn_c > 0:
            f1_sum += f1
            valid_labels += 1
    macro_f1 = f1_sum / valid_labels if valid_labels else 0.0

    n_total = len(prepared)
    summary = {
        "checkpoint": str(ckpt_path),
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": n_correct / max(n_total, 1),
        "accuracy_parsed_only": n_correct / max(n_parsed, 1),
        "macro_f1": macro_f1,
        "per_source_accuracy": {
            src: {
                "n": by_source_total[src],
                "parsed": by_source_parsed[src],
                "correct": by_source_correct[src],
                "acc": by_source_correct[src] / max(by_source_total[src], 1),
            }
            for src in sorted(by_source_total)
        },
        "per_label_f1": per_label,
        "elapsed_sec": time.time() - t0,
        "eval_stem": "native" if use_native_question else EVAL_EMOTION_STEM,
        "max_audio_samples": max_audio_samples,
        "use_cache": use_cache,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[listen] {ckpt_path.name} "
          f"acc={summary['accuracy']:.4f} macroF1={macro_f1:.4f} "
          f"parsed={n_parsed}/{n_total}", flush=True)

    # Free GPU mem
    del model
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True, help="Dir containing checkpoint-*")
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None,
                   help="Base model dir for adapter-only ckpts (default: read from adapter_config.json)")
    p.add_argument("--ckpts", default=None,
                   help="Comma-separated step numbers to include (default: all)")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap rows (debug/smoke)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Truncate waveform above this. Default: 1_600_000 (DAC) / "
                        "480_000 (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--use-native-question", action="store_true",
                   help="Use LISTEN's own question field instead of the canonical "
                        "TASK_PROMPTS emotion_classify stem.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable KV/conv cache during generation — slower, but "
                        "bypasses the Qwen3Next cache-API shims in _loader.")
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
    ckpts = find_checkpoints(
        ckpt_root, steps_filter,
        min_age_sec=0 if args.include_partial else 60,
    )
    if not ckpts:
        raise SystemExit(f"[listen] no checkpoints under {ckpt_root}")

    print(f"[listen] loading {args.parquet}", flush=True)
    rows = load_listen_test(args.parquet)
    print(f"[listen] {len(rows)} rows", flush=True)

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            print(f"[listen] {p.name} already done", flush=True)
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue
        try:
            summary = eval_checkpoint(
                p, args.base_model, rows,
                batch_size=args.batch_size,
                out_dir=out_dir,
                use_native_question=args.use_native_question,
                max_samples=args.max_samples,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_cache,
            )
            all_summaries.append(summary)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'checkpoint':<24s}  {'acc':>6s}  {'macroF1':>8s}  {'parsed':>12s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        print(f"{name:<24s}  {s['accuracy']:>6.4f}  {s['macro_f1']:>8.4f}  "
              f"{s['n_parsed']:>5d}/{s['n_total']:<5d}")


if __name__ == "__main__":
    main()
