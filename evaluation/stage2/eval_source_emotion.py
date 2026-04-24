"""Source-corpus emotion eval for Qwen3.5AE Stage-2 checkpoints.

Evaluates on the PER-CORPUS held-out splits that Stage-2 training explicitly
reserved (see [`stage2_eval_plan.md §7.1`](../docs/stage2_eval_plan.md) and
[`build_training_manifest.py`](../scripts/emo/build_training_manifest.py)):

    MELD        : official test split          (2 747 wavs, 7-class)
    DailyTalk   : last 5% of dialogues         (~1 168 dialogs, 7-class)
    EmoV-DB     : Jenie speaker                (1 790 wavs, 5-class)
    RAVDESS     : Actors 21-24                 (240 wavs, 8-class)

Prompt format mirrors training's emotion MCQA rows (lettered choices, same
`"Answer with the letter."` suffix) so the training distribution covers the
eval phrasing. Per-corpus metrics: accuracy, macro-F1, balanced-accuracy,
per-class accuracy.

Usage:
    python -m evaluation.stage2.eval_source_emotion \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_source_emotion \
        --base-model /mnt/tmp/s2_init_42k \
        --ckpts 1000,2000 --corpora meld dailytalk emov ravdess --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
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

RAW = Path("/mnt/tmp/datasets/emotion_raw")

# Same canonical stem as TASK_PROMPTS["emotion_classify"][0] and omni_dataset
# training. Pinned here to avoid cross-module import dep.
QUESTION = "What emotion does the speaker convey?"

# Per-corpus taxonomies: identical to scripts/emo/prepare_emotion_mcqa_manifest.py
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
DAILYTALK_EMOTIONS = ["no emotion", "happiness", "sadness", "anger", "surprise",
                     "fear", "disgust"]
EMOV_EMOTIONS = ["amused", "angry", "disgusted", "neutral", "sleepy"]
RAVDESS_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful",
                    "disgust", "surprised"]

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
MAX_NEW_TOKENS = 96  # letter + optional rationale (training target may include one)


# ---------------------------------------------------------------------------
# per-corpus held-out split enumeration
# ---------------------------------------------------------------------------


def load_meld_test() -> list[dict]:
    csvp = RAW / "MELD" / "MELD.Raw" / "test_sent_emo.csv"
    audio_dir = RAW / "MELD" / "audio" / "test"
    df = pd.read_csv(csvp)
    rows = []
    for _, r in df.iterrows():
        stem = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
        ap = audio_dir / f"{stem}.wav"
        if not ap.exists():
            continue
        rows.append({
            "id": stem,
            "path": str(ap),
            "label": str(r["Emotion"]).strip().lower(),
            "utterance": str(r["Utterance"]),
        })
    return rows


def load_dailytalk_heldout() -> list[dict]:
    """Last 5% of dialogues (matches build_training_manifest.py:136-148)."""
    meta = json.loads((RAW / "DailyTalk" / "dailytalk" / "metadata.json").read_text())
    d_root = RAW / "DailyTalk" / "dailytalk" / "data"
    ids_sorted = sorted(int(k) for k in meta.keys())
    if not ids_sorted:
        return []
    cutoff = ids_sorted[int(len(ids_sorted) * 0.95)]
    held = [i for i in ids_sorted if i >= cutoff]

    rows = []
    for did in held:
        dlg_dir = d_root / str(did)
        if not dlg_dir.exists():
            continue
        for wav_path in sorted(dlg_dir.glob("*.wav")):
            stem = wav_path.stem
            # filename "{utt_id}_{speaker}_d{dialog_id}.wav"
            m = re.match(r"(\d+)_(\d+)_d(\d+)", stem)
            if not m:
                continue
            utt_id, _, dlg_id = m.group(1), m.group(2), m.group(3)
            dlg_meta = meta.get(dlg_id)
            if dlg_meta is None:
                continue
            utt_meta = dlg_meta.get(utt_id)
            if utt_meta is None:
                continue
            rows.append({
                "id": stem,
                "path": str(wav_path),
                "label": str(utt_meta.get("emotion", "no emotion")).strip().lower(),
                "utterance": utt_meta.get("text", ""),
            })
    return rows


def load_emov_jenie() -> list[dict]:
    d = RAW / "EmoV-DB" / "jenie"
    if not d.exists():
        return []
    rows = []
    for wav_path in sorted(d.rglob("*.wav")):
        emotion_dir = wav_path.parent.name.lower()  # Amused / Angry / …
        if emotion_dir not in {"amused", "angry", "disgusted", "neutral", "sleepy"}:
            continue
        rows.append({
            "id": wav_path.stem,
            "path": str(wav_path),
            "label": emotion_dir,
            "utterance": "",
        })
    return rows


_RAV_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")

def load_ravdess_heldout() -> list[dict]:
    """Actors 21-24 (standard RAVDESS speaker-held-out split)."""
    d = RAW / "RAVDESS"
    rows = []
    for actor_n in (21, 22, 23, 24):
        actor_dir = d / f"Actor_{actor_n:02d}"
        if not actor_dir.exists():
            continue
        for wav_path in sorted(actor_dir.glob("*.wav")):
            m = _RAV_RE.search(wav_path.stem)
            if not m:
                continue
            emo_idx = int(m.group(3))
            if not (1 <= emo_idx <= 8):
                continue
            rows.append({
                "id": wav_path.stem,
                "path": str(wav_path),
                "label": RAVDESS_EMOTIONS[emo_idx - 1],
                "utterance": "",
            })
    return rows


CORPUS_LOADERS = {
    "meld":      (load_meld_test,        MELD_EMOTIONS),
    "dailytalk": (load_dailytalk_heldout, DAILYTALK_EMOTIONS),
    "emov":      (load_emov_jenie,       EMOV_EMOTIONS),
    "ravdess":   (load_ravdess_heldout,  RAVDESS_EMOTIONS),
}


# ---------------------------------------------------------------------------
# prompt + parsing (matches omni_dataset.py emotion branch)
# ---------------------------------------------------------------------------


def build_user_suffix(choices: list[str]) -> str:
    """Produce the training-identical user suffix.

    omni_dataset builds: f"{stem}\nChoices: {' '.join(lettered_choices)}\nAnswer with the letter."
    """
    lettered = [f"{LETTERS[i]}) {c}" for i, c in enumerate(choices)]
    return f"{QUESTION}\nChoices: {' '.join(lettered)}\nAnswer with the letter."


_LETTER_RE = re.compile(r"\b([A-H])\b")


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


def decode_audio(path: str, max_samples: int) -> torch.Tensor:
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
# run one corpus
# ---------------------------------------------------------------------------


def run_corpus(
    model, tokenizer, cfg,
    corpus: str,
    rows: list[dict],
    choices: list[str],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool,
    ckpt_path: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{corpus}.jsonl"
    summary_path = out_dir / f"summary_{corpus}.json"

    choice_lower = [c.lower() for c in choices]

    # Pre-decode audio once; length-sort for padding efficiency.
    print(f"[src-emo] {corpus}: decoding {len(rows)} audios...", flush=True)
    prepared = []
    skip_label = Counter()
    for r in rows:
        if r["label"] not in choice_lower:
            skip_label[r["label"]] += 1
            continue
        try:
            wav = decode_audio(r["path"], max_audio_samples)
        except Exception as e:
            print(f"[src-emo] {corpus}: decode fail {r['id']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav, "_gold_idx": choice_lower.index(r["label"])})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    if skip_label:
        print(f"[src-emo] {corpus}: label-out-of-taxonomy skipped: {dict(skip_label)}",
              flush=True)

    audio_pad_id = cfg.audio_pad_token_id
    user_suffix = build_user_suffix(choices)  # identical across rows within a corpus
    suffix_ids_unused = tokenizer.encode(user_suffix, add_special_tokens=False)

    n_correct = 0
    n_parsed = 0
    conf_true: list[str] = []
    conf_pred: list[str] = []

    t0 = time.time()
    with open(pred_path, "w") as fp:
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            prompts, waveforms = [], []
            for r in batch:
                t_audio = max(1, r["_wav"].shape[-1] // HOP_LENGTH)
                prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, user_suffix))
                waveforms.append(r["_wav"])
            try:
                raws = generate_greedy(
                    model, tokenizer, cfg, prompts, waveforms,
                    max_new_tokens=max_new_tokens, use_cache=use_cache,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raws = []
                for j, p in enumerate(prompts):
                    raws.extend(generate_greedy(
                        model, tokenizer, cfg, [p], [waveforms[j]],
                        max_new_tokens=max_new_tokens, use_cache=use_cache,
                    ))

            for r, raw in zip(batch, raws):
                pred_letter = parse_letter(raw, len(choices))
                gold_letter = LETTERS[r["_gold_idx"]]
                pred_label = (choices[ord(pred_letter) - ord("A")]
                              if pred_letter is not None else None)
                correct = pred_letter == gold_letter

                if pred_letter is not None:
                    n_parsed += 1
                if correct:
                    n_correct += 1
                conf_true.append(r["label"])
                conf_pred.append(pred_label or "")

                fp.write(json.dumps({
                    "id": r["id"],
                    "path": r["path"],
                    "gold_label": r["label"],
                    "gold_letter": gold_letter,
                    "pred_letter": pred_letter,
                    "pred_label": pred_label,
                    "raw_output": raw,
                    "correct": correct,
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                print(f"[src-emo] {ckpt_path.name}/{corpus} {done}/{len(prepared)}  "
                      f"acc={n_correct/max(done,1):.3f} parsed={n_parsed}/{done}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    # Per-class accuracy + macro-F1
    per_class = {}
    f1_sum = 0.0
    n_labels_with_support = 0
    for lbl in choice_lower:
        support = sum(1 for t in conf_true if t == lbl)
        if support == 0:
            continue
        tp = sum(1 for t, p in zip(conf_true, conf_pred) if t == lbl and p == lbl)
        fp_c = sum(1 for t, p in zip(conf_true, conf_pred) if t != lbl and p == lbl)
        fn_c = sum(1 for t, p in zip(conf_true, conf_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp_c) if (tp + fp_c) else 0.0
        rec = tp / (tp + fn_c) if (tp + fn_c) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {
            "support": support,
            "accuracy": tp / support,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        f1_sum += f1
        n_labels_with_support += 1

    macro_f1 = f1_sum / max(n_labels_with_support, 1)
    balanced_acc = sum(per_class[l]["recall"] for l in per_class) / max(
        n_labels_with_support, 1)

    total = len(prepared)
    summary = {
        "checkpoint": str(ckpt_path),
        "corpus": corpus,
        "taxonomy": choices,
        "n_total": total,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": n_correct / max(total, 1),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "random_baseline_n_class": 1 / len(choices),
        "elapsed_sec": time.time() - t0,
        "use_cache": use_cache,
        "skipped_label_out_of_taxonomy": dict(skip_label),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[src-emo] {ckpt_path.name}/{corpus}  "
          f"acc={summary['accuracy']:.4f} macroF1={macro_f1:.4f} "
          f"bal_acc={balanced_acc:.4f}  n={total} "
          f"(random={1/len(choices):.3f})", flush=True)
    return summary


# ---------------------------------------------------------------------------
# per-ckpt driver
# ---------------------------------------------------------------------------


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    corpora: list[str],
    data: dict[str, tuple[list[dict], list[str]]],
    batch_size: int,
    out_dir: Path,
    max_samples: int | None,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)

    summaries: dict[str, dict] = {}
    for corpus in corpora:
        rows, choices = data[corpus]
        if max_samples is not None and max_samples < len(rows):
            rows = rows[:max_samples]
        s = run_corpus(
            model, tokenizer, cfg, corpus, rows, choices,
            batch_size=batch_size,
            out_dir=out_dir,
            max_audio_samples=max_audio_samples,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            ckpt_path=ckpt_path,
        )
        summaries[corpus] = s

    combined = {"checkpoint": str(ckpt_path), "per_corpus": summaries}
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
    p.add_argument("--corpora", nargs="+",
                   choices=list(CORPUS_LOADERS.keys()),
                   default=list(CORPUS_LOADERS.keys()))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=1_600_000)
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
        raise SystemExit(f"[src-emo] no checkpoints under {ckpt_root}")

    # Pre-load all held-out splits once
    data: dict[str, tuple[list[dict], list[str]]] = {}
    for corpus in args.corpora:
        loader, choices = CORPUS_LOADERS[corpus]
        rows = loader()
        print(f"[src-emo] {corpus:<10s}  loaded {len(rows)} rows "
              f"(taxonomy {len(choices)}-class: {choices})", flush=True)
        data[corpus] = (rows, choices)

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        if (out_dir / "summary.json").exists():
            with open(out_dir / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, args.corpora, data,
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

    print("\n=== SUMMARY (source-corpus held-out emotion eval) ===")
    print(f"{'ckpt':<16s} {'corpus':<10s} {'n_class':>7s}  {'n':>5s}  "
          f"{'acc':>6s}  {'macroF1':>7s}  {'bal_acc':>7s}  {'random':>7s}")
    for combo in all_summaries:
        name = Path(combo["checkpoint"]).name
        for corpus, s in combo["per_corpus"].items():
            print(f"{name:<16s} {corpus:<10s} {len(s['taxonomy']):>7d}  "
                  f"{s['n_total']:>5d}  "
                  f"{s['accuracy']:>6.4f}  {s['macro_f1']:>7.4f}  "
                  f"{s['balanced_accuracy']:>7.4f}  {s['random_baseline_n_class']:>7.4f}")


if __name__ == "__main__":
    main()
