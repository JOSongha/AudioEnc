"""MSP-Podcast emotion eval (4-class subset: angry/happy/neutral/sad).

Source: CLAPv2/MSP_podcast HF mirror (11 parquets, ~27k+ rows after primary-class filter).
Schema: image (audio bytes, mislabeled), arousal, valence, label, arousal_norm, valence_norm.
Label mapping (inferred from priority-order distribution, no README):
    0 = neutral, 1 = happy, 2 = sad, 3 = angry, 4 = surprise, 5 = disgust
This eval uses 0-3 (4-class) only, subsampled to MAX_PER_CLASS for tractability.

NOT in any v3/v4/Stage-2 training pool.

Usage:
    python -m evaluation.stage2.eval_msp_podcast \\
        --ckpt-root /mnt/tmp/v4_ckpts_only \\
        --out-root  /mnt/tmp/.../eval_msp_podcast \\
        --ckpts 68000

Add --base-model for Stage-2 LoRA.
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

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

CACHE_ROOT = Path("/mnt/tmp/cache/CLAPv2_MSP_podcast")

# Priority-order assumption (verify by sanity check on small sample first)
LABELS = ["neutral", "happy", "sad", "angry"]  # int 0,1,2,3
CHOICES = [f"{chr(ord('A')+i)}. {l}" for i, l in enumerate(LABELS)]
LETTER_BY_LABEL = {l: chr(ord('A')+i) for i, l in enumerate(LABELS)}
LABEL_BY_LETTER = {chr(ord('A')+i): l for i, l in enumerate(LABELS)}
QUESTION = "What is the emotion expressed?"

MAX_PER_CLASS = 400  # balanced subsample (4×400 = 1600 max)


def load_msp(seed: int = 42) -> list[dict]:
    """Download all 11 parquets, filter labels 0-3, balanced subsample."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi()
    files = sorted(f for f in api.list_repo_files('CLAPv2/MSP_podcast', repo_type='dataset') if f.endswith('.parquet'))
    print(f"[msp] {len(files)} parquets")
    by_label = defaultdict(list)
    for f in files:
        p = hf_hub_download(repo_id='CLAPv2/MSP_podcast', repo_type='dataset',
                            filename=f, local_dir=str(CACHE_ROOT))
        t = pq.read_table(p)
        df = t.to_pandas()
        for _, r in df.iterrows():
            lab = int(r["label"])
            if lab not in (0, 1, 2, 3):
                continue
            audio = r["image"]  # mislabeled as image
            ab = audio.get("bytes") if isinstance(audio, dict) else None
            if not ab:
                continue
            by_label[lab].append({
                "id": f"msp_{lab}_{len(by_label[lab])}",
                "audio_bytes": ab,
                "gold_label": LABELS[lab],
                "gold_letter": LETTER_BY_LABEL[LABELS[lab]],
            })
    rng = random.Random(seed)
    rows = []
    for lab in sorted(by_label):
        pool = by_label[lab]
        rng.shuffle(pool)
        rows.extend(pool[:MAX_PER_CLASS])
    rng.shuffle(rows)
    print(f"[msp] balanced subsample: total={len(rows)}, dist={dict(Counter(r['gold_label'] for r in rows))}")
    return rows


def preprocess_audio(audio_bytes: bytes, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).to(torch.float32)


def build_prompt(tokenizer, audio_pad_id, t_audio):
    choices_str = "\n".join(CHOICES)
    user_suffix = f"{QUESTION}\n{choices_str}\nAnswer with the letter."
    return build_prompt_ids(tokenizer, audio_pad_id, t_audio, user_suffix)


@torch.inference_mode()
def run_eval(model, tokenizer, cfg, rows, *, batch_size, max_audio_samples, max_new_tokens):
    target_sr = audio_sample_rate(cfg)
    audio_pad_id = cfg.audio_pad_token_id
    prepared = []
    for r in rows:
        wav = preprocess_audio(r["audio_bytes"], target_sr)
        if wav.numel() > max_audio_samples:
            wav = wav[:max_audio_samples]
        prepared.append({**r, "wav": wav})
    prepared.sort(key=lambda x: x["wav"].numel())
    preds = []
    t0 = time.time()
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        prompts, wavs = [], []
        for b in batch:
            t_a = t_audio_for(cfg, b["wav"].numel())
            prompts.append(build_prompt(tokenizer, audio_pad_id, t_a))
            wavs.append(b["wav"])
        gen = generate_greedy(model, tokenizer, cfg, prompts, wavs,
                              max_new_tokens=max_new_tokens, use_cache=True)
        for b, txt in zip(batch, gen):
            m = re.search(r"[A-D]", txt or "")
            pl = m.group(0) if m else ""
            preds.append({
                "id": b["id"], "gold_label": b["gold_label"],
                "gold_letter": b["gold_letter"], "pred_letter": pl,
                "pred_label": LABEL_BY_LETTER.get(pl, ""),
                "raw_output": txt,
                "correct": pl == b["gold_letter"],
            })
        if (start // batch_size) % 20 == 0:
            done = start + len(batch)
            dt = time.time() - t0
            rate = done / max(dt, 1e-6)
            eta = (len(prepared) - done) / max(rate, 1e-6)
            print(f"[msp] {done}/{len(prepared)}  {rate:.2f} sps  eta {eta:.0f}s", flush=True)
    return preds


def summarise(preds, ckpt_path):
    n = len(preds)
    correct = sum(1 for p in preds if p["correct"])
    sup = Counter(p["gold_label"] for p in preds)
    cm = defaultdict(lambda: Counter())
    for p in preds:
        cm[p["gold_label"]][p["pred_label"] or "<empty>"] += 1
    pc = {}
    for lab in LABELS:
        s = sup[lab]
        tp = cm[lab][lab]
        fp = sum(cm[g][lab] for g in cm if g != lab)
        rec = tp / s if s else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        pc[lab] = {"support": s, "recall": rec, "precision": prec, "f1": f1}
    macro_f1 = sum(pc[l]["f1"] for l in pc) / len(pc)
    wa = sum(pc[l]["recall"] * sup[l] for l in pc) / max(n, 1)
    return {
        "checkpoint": str(ckpt_path),
        "n_total": n, "accuracy": correct / n if n else 0,
        "macro_f1": macro_f1, "weighted_accuracy": wa,
        "support": dict(sup), "per_class": pc,
        "confusion_matrix": {g: dict(cm[g]) for g in cm},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--ckpts", type=str, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-audio-samples", type=int, default=None)
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    args = p.parse_args()

    rows = load_msp()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    steps = {int(s) for s in args.ckpts.split(",")} if args.ckpts else None
    ckpts = find_checkpoints(Path(args.ckpt_root), steps_filter=steps)
    print(f"[msp] {len(ckpts)} checkpoint(s)", flush=True)
    for ckpt_path in ckpts:
        out_dir = out_root / ckpt_path.name
        sum_path = out_dir / "summary.json"
        if sum_path.exists():
            print(f"[msp] skip {ckpt_path.name}", flush=True); continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[msp] loading {ckpt_path}", flush=True)
        model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=args.base_model,
                                                attn_implementation=args.attn_implementation)
        max_audio = args.max_audio_samples or default_max_audio_samples(cfg)
        preds = run_eval(model, tokenizer, cfg, rows, batch_size=args.batch_size,
                         max_audio_samples=max_audio, max_new_tokens=args.max_new_tokens)
        with open(out_dir / "predictions.jsonl", "w") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarise(preds, ckpt_path)
        with open(sum_path, "w") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"[msp] {ckpt_path.name}  acc={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}  WA={s['weighted_accuracy']:.4f}  n={s['n_total']}", flush=True)
        del model, tokenizer; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
