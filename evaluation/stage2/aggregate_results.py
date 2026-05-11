"""Aggregate Stage-2 eval results into a CSV / markdown table.

Walks every eval_*/checkpoint-N/summary.json under a results dir, extracts
the representative metric(s) per (eval, ckpt), and emits:

    <out>/results.csv             long format: ckpt,eval,metric,value
    <out>/results_wide.csv        wide format: ckpt × eval-metric matrix
    <out>/results.md              markdown table (paper-ready)
    <out>/best_per_task.md        markdown summary of best ckpt per task

Also feeds plot_trajectories.py.

Usage:
    python -m evaluation.stage2.aggregate_results \
        --root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out  .../analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Per-eval-dir → list of (column_label, callable extracting the metric).
# Lower-is-better metrics flagged with `lower=True` for best-ckpt logic.
EVAL_METRICS: dict[str, list[tuple[str, callable, bool]]] = {
    "eval_source_emotion": [
        ("MELD_acc",       lambda s: s["per_corpus"]["meld"]["accuracy"], False),
        ("MELD_F1",        lambda s: s["per_corpus"]["meld"]["macro_f1"], False),
        ("DailyTalk_acc",  lambda s: s["per_corpus"]["dailytalk"]["accuracy"], False),
        ("DailyTalk_F1",   lambda s: s["per_corpus"]["dailytalk"]["macro_f1"], False),
        ("EmoV_acc",       lambda s: s["per_corpus"]["emov"]["accuracy"], False),
        ("EmoV_F1",        lambda s: s["per_corpus"]["emov"]["macro_f1"], False),
        ("RAVDESS_acc",    lambda s: s["per_corpus"]["ravdess"]["accuracy"], False),
        ("RAVDESS_F1",     lambda s: s["per_corpus"]["ravdess"]["macro_f1"], False),
    ],
    "eval_esc50": [
        ("ESC50_acc", lambda s: s["accuracy_pooled"], False),
    ],
    "eval_esc50_acc": [
        ("ESC50_acc", lambda s: s["accuracy_pooled"], False),
        ("ESC50_acc_per_fold_mean", lambda s: s.get("accuracy_mean_per_fold"), False),
    ],
    "eval_clotho": [
        ("Clotho_BLEU1", lambda s: s["bleu1"], False),
        ("Clotho_BLEU4", lambda s: s["bleu4"], False),
        ("Clotho_CIDEr", lambda s: s.get("CIDEr"), False),
        ("Clotho_METEOR", lambda s: s.get("METEOR"), False),
    ],
    "eval_clotho_caption": [
        ("Clotho_BLEU1", lambda s: s["bleu1"], False),
        ("Clotho_BLEU4", lambda s: s["bleu4"], False),
        ("Clotho_CIDEr", lambda s: s.get("CIDEr"), False),
        ("Clotho_METEOR", lambda s: s.get("METEOR"), False),
        ("Clotho_ROUGE_L", lambda s: s.get("ROUGE_L"), False),
        ("Clotho_SPICE", lambda s: s.get("SPICE"), False),
    ],
    "eval_fsd50k": [
        ("FSD50K_F1mi",   lambda s: s["f1_micro"], False),
        ("FSD50K_F1ma",   lambda s: s["f1_macro"], False),
        ("FSD50K_Jacc",   lambda s: s["jaccard_mean"], False),
    ],
    "eval_fsd50k_map": [
        ("FSD50K_F1mi",   lambda s: s["f1_micro"], False),
        ("FSD50K_F1ma",   lambda s: s["f1_macro"], False),
        ("FSD50K_Pmi",    lambda s: s.get("precision_micro"), False),
        ("FSD50K_Rmi",    lambda s: s.get("recall_micro"), False),
        ("FSD50K_Jacc",   lambda s: s["jaccard_mean"], False),
    ],
    "eval_fsd50k_map_seq": [
        ("FSD50K_mAPma",  lambda s: s.get("mAP_macro"), False),
        ("FSD50K_mAPmi",  lambda s: s.get("mAP_micro"), False),
    ],
    "eval_audioset_map": [
        ("AudioSet_F1mi", lambda s: s["f1_micro"], False),
        ("AudioSet_F1ma", lambda s: s["f1_macro"], False),
        ("AudioSet_Jacc", lambda s: s["jaccard"], False),
    ],
    "eval_librispeech": [
        ("WER_clean", lambda s: s.get("wer_normalized", s.get("wer")), True),
        ("CER_clean", lambda s: s.get("cer_normalized", s.get("cer")), True),
    ],
    "eval_librispeech_wer": [
        ("WER_clean", lambda s: s.get("wer_normalized", s.get("wer")), True),
        ("CER_clean", lambda s: s.get("cer_normalized", s.get("cer")), True),
    ],
    "eval_librispeech_other": [
        ("WER_other", lambda s: s.get("wer_normalized", s.get("wer")), True),
        ("CER_other", lambda s: s.get("cer_normalized", s.get("cer")), True),
    ],
    "eval_librispeech_wer_other": [
        ("WER_other", lambda s: s.get("wer_normalized", s.get("wer")), True),
        ("CER_other", lambda s: s.get("cer_normalized", s.get("cer")), True),
    ],
    "eval_listen": [
        ("LISTEN_acc",     lambda s: s["accuracy"], False),
        ("LISTEN_F1",      lambda s: s["macro_f1"], False),
    ],
    "eval_listen_mcqa": [
        ("LISTEN_acc",     lambda s: s["accuracy"], False),
        ("LISTEN_F1",      lambda s: s["macro_f1"], False),
        ("LISTEN_acc_parsed", lambda s: s.get("accuracy_parsed_only"), False),
    ],
    "eval_listen_official": [
        # Per-experiment metric. Aggregate by mean across experiments.
        ("LISTENo_WAmean", lambda s: _listen_official_mean(s, "weighted_accuracy"), False),
        ("LISTENo_F1mean", lambda s: _listen_official_mean(s, "macro_f1"), False),
    ],
    "eval_asr_external": [
        ("MLS_WER",       lambda s: _asr_ext(s, "mls", "wer_normalized"), True),
        ("MLS_CER",       lambda s: _asr_ext(s, "mls", "cer_normalized"), True),
        ("VoxPopuli_WER", lambda s: _asr_ext(s, "voxpopuli", "wer_normalized"), True),
        ("VoxPopuli_CER", lambda s: _asr_ext(s, "voxpopuli", "cer_normalized"), True),
        ("Gigaspeech_WER", lambda s: _asr_ext(s, "gigaspeech", "wer_normalized"), True),
        ("Gigaspeech_CER", lambda s: _asr_ext(s, "gigaspeech", "cer_normalized"), True),
    ],
}


def _asr_ext(s: dict, ds: str, key: str):
    d = s.get("datasets", {}).get(ds, {})
    if d.get("status") == "ok":
        return d.get(key)
    return None


def _listen_official_mean(s: dict, key: str) -> float | None:
    pe = s.get("per_experiment", {})
    vals = [v.get(key) for v in pe.values()
            if isinstance(v, dict) and v.get("available") and v.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _ckpt_step(name: str) -> int | None:
    m = re.match(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for eval_dir, metric_list in EVAL_METRICS.items():
        d = root / eval_dir
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir(), key=lambda p: _ckpt_step(p.name) or 99999):
            step = _ckpt_step(sub.name)
            if step is None:
                continue
            sf = sub / "summary.json"
            if not sf.exists():
                continue
            try:
                s = json.loads(sf.read_text())
            except Exception:
                continue
            for col, fn, lower in metric_list:
                try:
                    v = fn(s)
                except (KeyError, TypeError):
                    v = None
                if v is None:
                    continue
                rows.append({"ckpt": step, "eval": eval_dir, "metric": col,
                             "value": float(v), "lower_is_better": lower})
    return rows


def write_long_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "eval", "metric", "value", "lower_is_better"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["ckpt"], r["metric"])):
            w.writerow(r)


def write_wide_csv(rows: list[dict], path: Path) -> None:
    ckpts = sorted({r["ckpt"] for r in rows})
    metrics = []
    seen = set()
    for r in sorted(rows, key=lambda r: (r["eval"], r["metric"])):
        m = r["metric"]
        if m not in seen:
            metrics.append(m)
            seen.add(m)
    table = {(r["ckpt"], r["metric"]): r["value"] for r in rows}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt"] + metrics)
        for c in ckpts:
            w.writerow([c] + [
                table.get((c, m), "") for m in metrics
            ])


def write_markdown(rows: list[dict], path: Path) -> None:
    ckpts = sorted({r["ckpt"] for r in rows})
    metrics = []
    seen = set()
    for r in sorted(rows, key=lambda r: (r["eval"], r["metric"])):
        m = r["metric"]
        if m not in seen:
            metrics.append(m)
            seen.add(m)
    table = {(r["ckpt"], r["metric"]): r["value"] for r in rows}
    lower = {r["metric"]: r["lower_is_better"] for r in rows}

    # Best per metric (across measured ckpts)
    best_step = {}
    best_val = {}
    for m in metrics:
        vals = [(c, table.get((c, m))) for c in ckpts if table.get((c, m)) is not None]
        if not vals:
            continue
        if lower[m]:
            best_step[m], best_val[m] = min(vals, key=lambda x: x[1])
        else:
            best_step[m], best_val[m] = max(vals, key=lambda x: x[1])

    lines = []
    lines.append("# Stage-2 trial v1 — full ckpt × metric matrix\n")
    lines.append(f"Auto-generated. Source: `aggregate_results.py`, runs read from "
                 f"`/mnt/tmp/results/.../eval_*/checkpoint-N/summary.json`.\n")
    # Header
    lines.append("| ckpt | " + " | ".join(metrics) + " |")
    lines.append("|---:" + "|---:" * len(metrics) + "|")
    for c in ckpts:
        cells = [f"{c//1000}k"]
        for m in metrics:
            v = table.get((c, m))
            if v is None:
                cells.append("—")
            else:
                star = "**" if best_step.get(m) == c else ""
                fmt = ".4f" if abs(v) < 10 else ".2f"
                cells.append(f"{star}{v:{fmt}}{star}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("**Best per metric (bolded above):**")
    lines.append("")
    lines.append("| metric | best ckpt | best value | direction |")
    lines.append("|---|---:|---:|---|")
    for m in metrics:
        if m not in best_step:
            continue
        d = "lower-is-better" if lower[m] else "higher-is-better"
        v = best_val[m]
        fmt = ".4f" if abs(v) < 10 else ".2f"
        lines.append(f"| {m} | {best_step[m]//1000}k | {v:{fmt}} | {d} |")

    path.write_text("\n".join(lines) + "\n")


def write_best_per_task(rows: list[dict], path: Path) -> None:
    metrics = []
    seen = set()
    for r in sorted(rows, key=lambda r: (r["eval"], r["metric"])):
        m = r["metric"]
        if m not in seen:
            metrics.append(m)
            seen.add(m)
    table = {(r["ckpt"], r["metric"]): r["value"] for r in rows}
    lower = {r["metric"]: r["lower_is_better"] for r in rows}
    ckpts = sorted({r["ckpt"] for r in rows})

    lines = ["# Best ckpt per task (Stage-2 v1)\n"]
    lines.append("Computed across all measured ckpts in the v1 run.\n")
    lines.append("| metric | direction | best ckpt | best value | ckpt-1k | ckpt-12k | ckpt-25k |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for m in metrics:
        vals = [(c, table.get((c, m))) for c in ckpts if table.get((c, m)) is not None]
        if not vals:
            continue
        d = "↓" if lower[m] else "↑"
        if lower[m]:
            best_c, best_v = min(vals, key=lambda x: x[1])
        else:
            best_c, best_v = max(vals, key=lambda x: x[1])
        v1 = table.get((1000, m))
        v12 = table.get((12000, m))
        v25 = table.get((25000, m))
        fmt_v = lambda v: f"{v:.4f}" if v is not None and abs(v) < 10 else (f"{v:.2f}" if v is not None else "—")
        lines.append(f"| {m} | {d} | {best_c//1000}k | {best_v:.4f} | {fmt_v(v1)} | {fmt_v(v12)} | {fmt_v(v25)} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Stage-2 run dir (parent of eval_*)")
    p.add_argument("--out", required=True,
                   help="Output dir for CSV/markdown")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = collect(root)
    if not rows:
        raise SystemExit(f"[aggregate] no summary.json found under {root}/eval_*")

    write_long_csv(rows, out / "results.csv")
    write_wide_csv(rows, out / "results_wide.csv")
    write_markdown(rows, out / "results.md")
    write_best_per_task(rows, out / "best_per_task.md")

    metrics = sorted({r["metric"] for r in rows})
    ckpts = sorted({r["ckpt"] for r in rows})
    print(f"[aggregate] {len(rows)} (ckpt × metric) rows")
    print(f"[aggregate]   ckpts: {len(ckpts)} ({ckpts[0]//1000}k … {ckpts[-1]//1000}k)")
    print(f"[aggregate]   metrics: {len(metrics)} ({', '.join(metrics[:6])}…)")
    print(f"[aggregate] wrote {out}/results.csv, results_wide.csv, results.md, best_per_task.md")


if __name__ == "__main__":
    main()
