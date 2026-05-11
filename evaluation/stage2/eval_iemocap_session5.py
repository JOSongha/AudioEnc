"""IEMOCAP Session 5 emotion MCQA eval (4-class).

Standard IEMOCAP held-out: Session 5 (speakers M5/F5).
Labels: ang / hap / neu / sad (exc merged into hap, common practice).
Same prompt + greedy decode as eval_source_emotion.py.

Usage:
    python -m evaluation.stage2.eval_iemocap_session5 \
        --ckpt-root /path/to/stage1_run \
        --out-root  /path/to/eval_iemocap_session5 \
        --ckpts 100000

For Stage-2 LoRA ckpts add `--base-model /path/to/stage2_init`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# flash_attn requires GLIBC_2.32 (__libc_single_threaded). On EL7 systems the
# symbol is provided by glibc_stub.so (built by install_env.sh). Load it via
# ctypes before any flash_attn import so the dynamic linker can resolve it.
_stub = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib", "glibc_stub.so")
if os.path.exists(_stub):
    ctypes.CDLL(_stub, mode=ctypes.RTLD_GLOBAL)

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

ROOT = Path("/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release")
SESSION = "Session5"

# Standard 4-class IEMOCAP eval label set, with excitement (exc) merged into happiness (hap).
LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",   # exc → hap merge (de facto standard)
    "neu": "neutral",
    "sad": "sad",
}
CHOICES = ["A. angry", "B. happy", "C. neutral", "D. sad"]
LETTER_BY_LABEL = {"angry": "A", "happy": "B", "neutral": "C", "sad": "D"}
LABEL_BY_LETTER = {"A": "angry", "B": "happy", "C": "neutral", "D": "sad"}
QUESTION = "What is the emotion expressed?"


def parse_emoeval_dir(eval_dir: Path) -> list[dict]:
    """Parse */EmoEvaluation/*.txt files for Session 5 (categorical eval lines).

    USE_NUBES=1 일 때 eval_dir 내용을 nubes 에서 fetch (label .txt 들 + 각
    utterance wav 의 nubes_path 생성).
    """
    from evaluation.stage2._nubes_loader import (
        USE_NUBES, NUBES_BASES, list_nubes_dir, fetch_nubes_text)
    pat = re.compile(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\w+)")
    rows = []
    if USE_NUBES:
        nubes_root = NUBES_BASES["iemocap"]["root"]
        emoeval_prefix = f"{nubes_root}{SESSION}/dialog/EmoEvaluation/"
        txt_paths = [p for p in list_nubes_dir(emoeval_prefix, suffix=".txt")]
    else:
        txt_paths = [str(p) for p in sorted(eval_dir.glob("*.txt"))]

    for txt_path in sorted(txt_paths):
        if USE_NUBES:
            content = fetch_nubes_text(txt_path)
            dialog = txt_path.split("/")[-1].rsplit(".", 1)[0]
        else:
            with open(txt_path) as f:
                content = f.read()
            dialog = Path(txt_path).stem
        for line in content.splitlines():
            m = pat.match(line.rstrip())
            if not m:
                continue
            _, _, utt_id, emo_short = m.groups()
            if emo_short not in LABEL_MAP:
                continue
            label = LABEL_MAP[emo_short]
            if USE_NUBES:
                wav_path = (f"{NUBES_BASES['iemocap']['root']}{SESSION}"
                            f"/sentences/wav/{dialog}/{utt_id}.wav")
            else:
                wav = ROOT / SESSION / "sentences" / "wav" / dialog / f"{utt_id}.wav"
                if not wav.exists():
                    continue
                wav_path = str(wav)
            rows.append({
                "id": utt_id,
                "path": wav_path,
                "gold_short": emo_short,
                "gold_label": label,
                "gold_letter": LETTER_BY_LABEL[label],
            })
    return rows


def preprocess_audio(path: str, target_sr: int) -> torch.Tensor:
    from evaluation.stage2._nubes_loader import USE_NUBES, fetch_nubes_audio_tensor
    if USE_NUBES and not str(path).startswith("/"):
        wav, sr = fetch_nubes_audio_tensor(str(path), target_sr=target_sr)
    else:
        wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).to(torch.float32)


def build_prompt(tokenizer, audio_pad_id: int, t_audio: int) -> list[int]:
    """Same MCQA format as eval_source_emotion.py / training rows."""
    choices_str = "\n".join(CHOICES)
    user_suffix = f"{QUESTION}\n{choices_str}\nAnswer with the letter."
    return build_prompt_ids(tokenizer, audio_pad_id, t_audio, user_suffix)


@torch.inference_mode()
def run_eval(model, tokenizer, cfg, rows, *, batch_size, max_audio_samples, max_new_tokens):
    target_sr = audio_sample_rate(cfg)
    audio_pad_id = cfg.audio_pad_token_id

    # preprocess
    prepared = []
    for r in rows:
        wav = preprocess_audio(r["path"], target_sr)
        if wav.numel() > max_audio_samples:
            wav = wav[:max_audio_samples]
        prepared.append({**r, "wav": wav})
    # sort by length for stable batches
    prepared.sort(key=lambda x: x["wav"].numel())

    preds = []
    t0 = time.time()
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        prompts = []
        wavs = []
        t_audios = []
        for b in batch:
            t_a = t_audio_for(cfg, b["wav"].numel())
            t_audios.append(t_a)
            prompts.append(build_prompt(tokenizer, audio_pad_id, t_a))
            wavs.append(b["wav"])
        gen_texts = generate_greedy(
            model, tokenizer, cfg, prompts, wavs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        for b, txt in zip(batch, gen_texts):
            m = re.search(r"[A-D]", txt or "")
            pred_letter = m.group(0) if m else ""
            pred_label = LABEL_BY_LETTER.get(pred_letter, "")
            preds.append({
                "id": b["id"],
                "path": b["path"],
                "gold_label": b["gold_label"],
                "gold_letter": b["gold_letter"],
                "pred_letter": pred_letter,
                "pred_label": pred_label,
                "raw_output": txt,
                "correct": pred_letter == b["gold_letter"],
            })
        if (start // batch_size) % 20 == 0:
            done = start + len(batch)
            dt = time.time() - t0
            rate = done / max(dt, 1e-6)
            eta = (len(prepared) - done) / max(rate, 1e-6)
            print(f"[iemocap-S5] {done}/{len(prepared)}  {rate:.2f} sps  eta {eta/60:.1f} min", flush=True)
    return preds


def summarise(preds, ckpt_path: Path) -> dict:
    n = len(preds)
    correct = sum(1 for p in preds if p["correct"])
    acc = correct / n if n else 0.0
    sup = Counter(p["gold_label"] for p in preds)
    cm = defaultdict(lambda: Counter())
    for p in preds:
        cm[p["gold_label"]][p["pred_label"] or "<empty>"] += 1
    per_class = {}
    for lab in ["angry", "happy", "neutral", "sad"]:
        s = sup[lab]
        tp = cm[lab][lab]
        fp = sum(cm[g][lab] for g in cm if g != lab)
        rec = tp / s if s else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_class[lab] = {
            "support": s, "recall": rec, "precision": prec, "f1": f1,
        }
    macro_f1 = sum(per_class[l]["f1"] for l in per_class) / len(per_class)
    weighted_acc = sum(per_class[l]["recall"] * sup[l] for l in per_class) / max(n, 1)
    return {
        "checkpoint": str(ckpt_path),
        "n_total": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_accuracy": weighted_acc,
        "support": dict(sup),
        "per_class": per_class,
        "confusion_matrix": {g: dict(cm[g]) for g in cm},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--ckpts", type=str, default=None,
                   help="comma-separated step numbers (e.g. 100000)")
    p.add_argument("--base-model", type=str, default=None,
                   help="base model dir for adapter-only (Stage-2 LoRA) ckpts")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-audio-samples", type=int, default=None)
    p.add_argument("--include-partial", action="store_true")
    p.add_argument("--attn-implementation", type=str, default="sdpa",
                   help="sdpa | eager | flash_attention_2")
    args = p.parse_args()

    eval_dir = ROOT / SESSION / "dialog" / "EmoEvaluation"
    rows = parse_emoeval_dir(eval_dir)
    print(f"[iemocap-S5] parsed {len(rows)} rows  label dist: {dict(Counter(r['gold_label'] for r in rows))}", flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    steps_filter = None
    if args.ckpts:
        steps_filter = {int(s) for s in args.ckpts.split(",")}
    ckpts = find_checkpoints(Path(args.ckpt_root), steps_filter=steps_filter)
    print(f"[iemocap-S5] {len(ckpts)} checkpoint(s) to evaluate", flush=True)

    for ckpt_path in ckpts:
        out_dir = out_root / ckpt_path.name
        sum_path = out_dir / "summary.json"
        if sum_path.exists():
            print(f"[iemocap-S5] skip {ckpt_path.name} (already done)", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[iemocap-S5] loading {ckpt_path}", flush=True)
        model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=args.base_model,
                                                attn_implementation=args.attn_implementation)
        max_audio_samples = args.max_audio_samples or default_max_audio_samples(cfg)
        preds = run_eval(model, tokenizer, cfg, rows,
                         batch_size=args.batch_size,
                         max_audio_samples=max_audio_samples,
                         max_new_tokens=args.max_new_tokens)
        with open(out_dir / "predictions.jsonl", "w") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarise(preds, ckpt_path)
        with open(sum_path, "w") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"[iemocap-S5] {ckpt_path.name}  acc={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}  WA={s['weighted_accuracy']:.4f}  n={s['n_total']}", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
