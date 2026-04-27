"""Tier-4 text-retention eval for Qwen3.5AE Stage-2 checkpoints.

Guards against LoRA corrupting the LLM backbone's general language ability.
Evaluates the 6 text-MCQA benchmarks Stage-2 training also used (train
splits); we score on the held-out validation/test splits downloaded at
/mnt/tmp/datasets/text_benchmarks/.

Schemas are rendered EXACTLY like prepare_text_sft_manifest.py so the
training prompt format covers the eval form; parse is first A..F letter.

Mapping (split / n_class):
    hellaswag       validation   4
    winogrande      validation   2
    boolq           validation   2
    arc_easy        test         4 (variable, often 4, sometimes 3 or 5)
    arc_challenge   test         4 (same)
    copa            validation   2

Usage:
    python -m evaluation.stage2.eval_text_retention \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_text_retention \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 1000,2000,5000,10000 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.stage2._loader import (  # noqa: E402
    build_prompt_ids,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
)

BENCH_ROOT = Path("/mnt/tmp/datasets/text_benchmarks")
LETTERS = ["A", "B", "C", "D", "E", "F"]
MAX_NEW_TOKENS = 8  # letter-only; keep short for throughput


# ---------------------------------------------------------------------------
# per-benchmark row builders (MIRROR prepare_text_sft_manifest.py)
# ---------------------------------------------------------------------------


def _arc_rows(bench: str, split: str) -> list[dict]:
    t = pq.read_table(BENCH_ROOT / bench / f"{split}.parquet")
    rows = []
    for i in range(t.num_rows):
        choices_struct = t.column("choices")[i].as_py()
        texts = list(choices_struct["text"])
        labels = list(choices_struct["label"])
        letter_map = {str(l): LETTERS[k] for k, l in enumerate(labels)}
        ans_raw = str(t.column("answerKey")[i].as_py())
        if ans_raw not in letter_map:
            continue
        rows.append({
            "id": str(t.column("id")[i].as_py()),
            "source": bench,
            "question": t.column("question")[i].as_py(),
            "choices": [f"{LETTERS[k]}) {c}" for k, c in enumerate(texts)],
            "answer": letter_map[ans_raw],
            "n_choices": len(texts),
        })
    return rows


def _winogrande_rows(split: str) -> list[dict]:
    t = pq.read_table(BENCH_ROOT / "winogrande" / f"{split}.parquet")
    rows = []
    for i in range(t.num_rows):
        sent = t.column("sentence")[i].as_py()
        opt1 = t.column("option1")[i].as_py()
        opt2 = t.column("option2")[i].as_py()
        ans = str(t.column("answer")[i].as_py())
        if ans not in ("1", "2"):
            continue
        rows.append({
            "id": f"winogrande_{i}",
            "source": "winogrande",
            "question": f'Fill in the blank: "{sent}"',
            "choices": [f"A) {opt1}", f"B) {opt2}"],
            "answer": "A" if ans == "1" else "B",
            "n_choices": 2,
        })
    return rows


def _hellaswag_rows(split: str) -> list[dict]:
    t = pq.read_table(BENCH_ROOT / "hellaswag" / f"{split}.parquet")
    rows = []
    for i in range(t.num_rows):
        ctx = t.column("ctx")[i].as_py()
        endings = list(t.column("endings")[i].as_py())
        try:
            ans = LETTERS[int(t.column("label")[i].as_py())]
        except (ValueError, IndexError, TypeError):
            continue
        rows.append({
            "id": f"hellaswag_{i}",
            "source": "hellaswag",
            "question": f'Which ending continues this passage most naturally?\n"{ctx}"',
            "choices": [f"{LETTERS[k]}) {e.strip()}" for k, e in enumerate(endings)],
            "answer": ans,
            "n_choices": len(endings),
        })
    return rows


def _boolq_rows(split: str) -> list[dict]:
    t = pq.read_table(BENCH_ROOT / "boolq" / f"{split}.parquet")
    rows = []
    for i in range(t.num_rows):
        q = (f'Read the passage and answer the question.\n'
             f'Passage: {t.column("passage")[i].as_py()}\n'
             f'Question: {t.column("question")[i].as_py()}')
        ans_raw = t.column("answer")[i].as_py()
        rows.append({
            "id": f"boolq_{i}",
            "source": "boolq",
            "question": q,
            "choices": ["A) Yes", "B) No"],
            "answer": "A" if bool(ans_raw) else "B",
            "n_choices": 2,
        })
    return rows


def _copa_rows(split: str) -> list[dict]:
    t = pq.read_table(BENCH_ROOT / "copa" / f"{split}.parquet")
    rows = []
    for i in range(t.num_rows):
        rel = "cause" if t.column("question")[i].as_py() == "cause" else "effect"
        q = f'{t.column("premise")[i].as_py()}\nWhat was the {rel}?'
        c1 = t.column("choice1")[i].as_py()
        c2 = t.column("choice2")[i].as_py()
        lbl = int(t.column("label")[i].as_py())
        rows.append({
            "id": f"copa_{i}",
            "source": "copa",
            "question": q,
            "choices": [f"A) {c1}", f"B) {c2}"],
            "answer": "A" if lbl == 0 else "B",
            "n_choices": 2,
        })
    return rows


BENCH_LOADERS = {
    "hellaswag":     ("validation", lambda: _hellaswag_rows("validation")),
    "winogrande":    ("validation", lambda: _winogrande_rows("validation")),
    "boolq":         ("validation", lambda: _boolq_rows("validation")),
    "arc_easy":      ("test",       lambda: _arc_rows("arc_easy", "test")),
    "arc_challenge": ("test",       lambda: _arc_rows("arc_challenge", "test")),
    "copa":          ("validation", lambda: _copa_rows("validation")),
}


# ---------------------------------------------------------------------------
# prompt + parse (matches omni_dataset.py text branch exactly)
# ---------------------------------------------------------------------------


def build_user_suffix(row: dict) -> str:
    """Mirrors omni_dataset._build_prompt_targets, modality="text":
        f"{q}\nChoices: {' '.join(choices)}\nAnswer with the letter."
    """
    return f"{row['question']}\nChoices: {' '.join(row['choices'])}\nAnswer with the letter."


_LETTER_RE = re.compile(r"\b([A-F])\b")


def parse_letter(text: str, n_choices: int) -> str | None:
    head = text.lstrip()[:20]
    m = _LETTER_RE.search(head)
    if m:
        c = m.group(1)
        if ord(c) - ord("A") < n_choices:
            return c
    m = _LETTER_RE.search(text)
    if m:
        c = m.group(1)
        if ord(c) - ord("A") < n_choices:
            return c
    return None


# ---------------------------------------------------------------------------
# per-benchmark runner
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_bench(
    model, tokenizer, cfg,
    bench: str,
    rows: list[dict],
    batch_size: int,
    out_dir: Path,
    max_new_tokens: int,
    use_cache: bool,
    ckpt_path: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{bench}.jsonl"
    summary_path = out_dir / f"summary_{bench}.json"

    audio_pad_id = cfg.audio_pad_token_id

    n_correct = 0
    n_parsed = 0
    pred_letter_counts = Counter()
    gold_letter_counts = Counter()
    by_letter_hit = Counter()
    by_letter_total = Counter()

    t0 = time.time()
    with open(pred_path, "w") as fp:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            prompts = [
                build_prompt_ids(tokenizer, audio_pad_id, 0, build_user_suffix(r))
                for r in batch
            ]
            try:
                raws = generate_greedy(
                    model, tokenizer, cfg, prompts, None,
                    max_new_tokens=max_new_tokens, use_cache=use_cache,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raws = []
                for p in prompts:
                    raws.extend(generate_greedy(
                        model, tokenizer, cfg, [p], None,
                        max_new_tokens=max_new_tokens, use_cache=use_cache,
                    ))

            for r, raw in zip(batch, raws):
                pred = parse_letter(raw, r["n_choices"])
                correct = (pred is not None and pred == r["answer"])
                if pred is not None:
                    n_parsed += 1
                    pred_letter_counts[pred] += 1
                gold_letter_counts[r["answer"]] += 1
                by_letter_total[r["answer"]] += 1
                if correct:
                    n_correct += 1
                    by_letter_hit[r["answer"]] += 1

                fp.write(json.dumps({
                    "id": r["id"],
                    "source": bench,
                    "n_choices": r["n_choices"],
                    "gold": r["answer"],
                    "pred": pred,
                    "raw": raw,
                    "correct": correct,
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(rows) - done) / max(rate, 1e-6)
                print(f"[text-ret] {ckpt_path.name}/{bench} {done}/{len(rows)}  "
                      f"acc={n_correct/max(done,1):.3f} parsed={n_parsed}/{done}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    total = len(rows)
    acc = n_correct / max(total, 1)

    # Per-gold-letter accuracy (catches letter-prior bias)
    per_letter_acc = {
        l: {"support": by_letter_total[l], "correct": by_letter_hit[l],
            "acc": by_letter_hit[l] / max(by_letter_total[l], 1)}
        for l in sorted(by_letter_total)
    }

    n_choices_dist = Counter(r["n_choices"] for r in rows)
    # Macro random baseline over choice cardinalities actually seen
    mixed_random = sum(n * (1 / k) for k, n in n_choices_dist.items()) / max(total, 1)

    summary = {
        "checkpoint": str(ckpt_path),
        "benchmark": bench,
        "n_total": total,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": acc,
        "random_baseline_mixed": mixed_random,
        "pred_letter_distribution": dict(pred_letter_counts),
        "gold_letter_distribution": dict(gold_letter_counts),
        "per_gold_letter": per_letter_acc,
        "n_choices_distribution": {str(k): v for k, v in n_choices_dist.items()},
        "elapsed_sec": time.time() - t0,
        "use_cache": use_cache,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[text-ret] {ckpt_path.name}/{bench}  "
          f"acc={acc:.4f} (random~{mixed_random:.3f})  parsed={n_parsed}/{total}",
          flush=True)
    return summary


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    benchmarks: list[str],
    data: dict[str, list[dict]],
    batch_size: int,
    out_dir: Path,
    max_new_tokens: int,
    use_cache: bool,
    max_samples: int | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)

    summaries: dict[str, dict] = {}
    for bench in benchmarks:
        rows = data[bench]
        if max_samples is not None and max_samples < len(rows):
            rows = rows[:max_samples]
        s = run_bench(
            model, tokenizer, cfg, bench, rows,
            batch_size=batch_size, out_dir=out_dir,
            max_new_tokens=max_new_tokens, use_cache=use_cache,
            ckpt_path=ckpt_path,
        )
        summaries[bench] = s

    # Unweighted mean across benchmarks (plan §7.3 aggregation rule)
    unweighted_mean = sum(s["accuracy"] for s in summaries.values()) / max(len(summaries), 1)
    combined = {
        "checkpoint": str(ckpt_path),
        "per_benchmark": summaries,
        "unweighted_mean_accuracy": unweighted_mean,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"[text-ret] {ckpt_path.name}  mean_acc={unweighted_mean:.4f}", flush=True)

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
    p.add_argument("--benchmarks", nargs="+", default=list(BENCH_LOADERS.keys()),
                   choices=list(BENCH_LOADERS.keys()))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap rows per benchmark (debug)")
    p.add_argument("--batch-size", type=int, default=8)
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
        raise SystemExit(f"[text-ret] no checkpoints under {ckpt_root}")

    data: dict[str, list[dict]] = {}
    for bench in args.benchmarks:
        split, loader = BENCH_LOADERS[bench]
        rows = loader()
        print(f"[text-ret] {bench:<14s}  split={split:<10s}  rows={len(rows)}", flush=True)
        data[bench] = rows

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        if (out_dir / "summary.json").exists():
            with open(out_dir / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, args.benchmarks, data,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_cache,
                max_samples=args.max_samples,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (Tier-4 text retention) ===")
    hdr = f"{'ckpt':<16s}  "
    for b in args.benchmarks:
        hdr += f"{b:>14s}  "
    hdr += f"{'mean':>8s}"
    print(hdr)
    for combo in all_summaries:
        name = Path(combo["checkpoint"]).name
        row = f"{name:<16s}  "
        for b in args.benchmarks:
            s = combo["per_benchmark"].get(b, {})
            row += f"{s.get('accuracy', float('nan')):>14.4f}  "
        row += f"{combo['unweighted_mean_accuracy']:>8.4f}"
        print(row)


if __name__ == "__main__":
    main()
