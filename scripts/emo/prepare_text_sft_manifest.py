"""Build the text-SFT manifest for Stage 2 from the 6 downloaded MCQA benchmark
train splits (ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, BoolQ, COPA).

Each row in the output JSONL is a unified MCQA row:
    {
      "source":    str,                 # benchmark name (e.g. "hellaswag")
      "question":  str,                 # fully-rendered question text
      "choices":   [str, ...],          # "A) ...", "B) ..." strings, ready to concat
      "answer":    str,                 # single letter (A/B/C/D/...)
      "modality":  "text"               # so the collator can detect text-only rows
    }

Subsample budget (total ≈ 6 100 to match plan 7:1 emotion:text with today's
emotion pool of 39.8 k — see stage2_listen_leakage_audit.md §9.1):
    HellaSwag   2 500
    WinoGrande  2 500
    ARC-Easy      400
    ARC-Challenge 400
    BoolQ         500
    COPA          400  (whole train split; ARC-C ≈ 400 is also whole)

Tier-4 eval guard: we train on train splits only; val / test splits stay held
out at /mnt/tmp/datasets/text_benchmarks/<name>/{validation,test}.parquet.
The "retention" interpretation of Tier 4 shifts from "does base Qwen's MMLU
hold?" to "did S2 LoRA + MCQA supervision raise OR degrade these six?" — see
stage2_eval_plan.md §7.3 caveat.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/tmp/datasets/text_benchmarks")
OUT = Path("/mnt/tmp/listen_analysis/train_manifest/text_sft_manifest.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(20260424)

# Budget per user-set 2026-04-24 plan: ASR:EMO:ENV:TXT = 0.83:1:1:0.5
# With today's emotion pool (~41 k), text target ≈ emotion/2 ≈ 20 500.
BUDGETS = {
    "hellaswag":     7500,
    "winogrande":    7500,
    "arc_easy":      2000,
    "arc_challenge": 1119,   # whole train split (max available)
    "boolq":         2000,
    "copa":           400,   # whole train split
}  # total ≈ 20 519

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ---- per-benchmark converters -------------------------------------------------
def conv_arc(df: pd.DataFrame, name: str) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        texts = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])          # may be A/B/C/D or 1/2/3/4
        # normalize to A/B/C/D
        letter_map = {}
        for i, lab in enumerate(labels):
            letter_map[str(lab)] = LETTERS[i]
        ans_raw = str(r["answerKey"])
        if ans_raw not in letter_map:
            continue   # malformed row
        out.append({
            "source": name,
            "question": r["question"],
            "choices": [f"{LETTERS[i]}) {t}" for i, t in enumerate(texts)],
            "answer": letter_map[ans_raw],
            "modality": "text",
        })
    return out


def conv_winogrande(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        q = f'Fill in the blank: "{r["sentence"]}"'
        choices = [f"A) {r['option1']}", f"B) {r['option2']}"]
        ans = "A" if str(r["answer"]) == "1" else "B"
        out.append({"source": "winogrande", "question": q, "choices": choices,
                    "answer": ans, "modality": "text"})
    return out


def conv_hellaswag(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        endings = list(r["endings"])
        q = f'Which ending continues this passage most naturally?\n"{r["ctx"]}"'
        choices = [f"{LETTERS[i]}) {e.strip()}" for i, e in enumerate(endings)]
        try:
            ans = LETTERS[int(r["label"])]
        except (ValueError, IndexError):
            continue
        out.append({"source": "hellaswag", "question": q, "choices": choices,
                    "answer": ans, "modality": "text"})
    return out


def conv_boolq(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        q = f'Read the passage and answer the question.\nPassage: {r["passage"]}\nQuestion: {r["question"]}'
        choices = ["A) Yes", "B) No"]
        ans = "A" if bool(r["answer"]) else "B"
        out.append({"source": "boolq", "question": q, "choices": choices,
                    "answer": ans, "modality": "text"})
    return out


def conv_copa(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        rel = "cause" if r["question"] == "cause" else "effect"
        q = f'{r["premise"]}\nWhat was the {rel}?'
        choices = [f"A) {r['choice1']}", f"B) {r['choice2']}"]
        ans = "A" if int(r["label"]) == 0 else "B"
        out.append({"source": "copa", "question": q, "choices": choices,
                    "answer": ans, "modality": "text"})
    return out


CONVERTERS = {
    "arc_easy":      lambda df: conv_arc(df, "arc_easy"),
    "arc_challenge": lambda df: conv_arc(df, "arc_challenge"),
    "winogrande":    conv_winogrande,
    "hellaswag":     conv_hellaswag,
    "boolq":         conv_boolq,
    "copa":          conv_copa,
}


def main() -> None:
    total_rows = []
    report = {}
    for name, budget in BUDGETS.items():
        p = ROOT / name / "train.parquet"
        if not p.exists():
            print(f"  {name:<15} MISSING — skip")
            report[name] = {"available": 0, "kept": 0}
            continue
        df = pd.read_parquet(p)
        rows = CONVERTERS[name](df)
        random.shuffle(rows)
        kept = rows[:budget]
        total_rows.extend(kept)
        report[name] = {"available": len(rows), "budget": budget, "kept": len(kept)}
        print(f"  {name:<15} avail={len(rows):>6}  budget={budget:>5}  kept={len(kept):>5}")

    random.shuffle(total_rows)
    with OUT.open("w") as f:
        for r in total_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT.parent / "text_sft_report.json").write_text(json.dumps(report, indent=2))

    print(f"\ntotal kept: {len(total_rows)}")
    print(f"manifest:   {OUT}")
    print(f"report:     {OUT.parent / 'text_sft_report.json'}")


if __name__ == "__main__":
    main()
