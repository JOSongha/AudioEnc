"""AudioSet eval-split sound-event multi-label eval for Qwen3.5AE Stage-2.

Mirrors `eval_fsd50k_map.py` but reads HuggingFace AudioSet parquet (embedded
FLAC bytes) and uses the AudioSet ontology (~527 labels appearing in eval).

The training manifest uses AudioSet `bal_train` (~18.7k clips); the eval split
(~17k clips) is a disjoint set of YouTube IDs — standard zero-shot AudioSet
multi-label classification.

Two scoring modes (same semantics as FSD50K):
  - `greedy`  : 1 forward/sample, parse comma-separated labels → F1/Jaccard.
  - `sequence`: teacher-forced log-prob per label → mAP (leaderboard-comparable),
                ~N_labels / label_batch_size forwards/sample.

Usage:
    python -m evaluation.stage2.eval_audioset_map \\
        --ckpt-root .../results/Qwen3.5AE-Stage2v2-... \\
        --out-root  .../eval_audioset \\
        --base-model /mnt/tmp/s2_init_42k \\
        --ckpts 11000 --batch-size 8 --score-mode greedy

    # or mAP (slow):
    --score-mode sequence --max-samples 200 --label-batch-size 50
"""

from __future__ import annotations

import argparse
import io
import json
import os as _os
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
    score_labels_teacher_forced,
    t_audio_for,
)

AUDIOSET_ROOT = Path("/mnt/tmp/datasets/env_sound/AudioSet")
EVAL_PARQUET_DIR = Path(_os.environ.get(
    "AUDIOSET_EVAL_PARQUET_DIR",
    str(AUDIOSET_ROOT / "data/eval"),
))
ONTOLOGY_JSON = AUDIOSET_ROOT / "ontology.json"

EVAL_STEM = "List the sound events in this audio, separated by commas."
SENTENCE_STEM = "Describe what you hear in this audio. Mention every distinct sound event."
MAX_NEW_TOKENS = 96
SENTENCE_MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_vocab() -> tuple[list[str], dict[str, int]]:
    """AudioSet vocab = label names from ontology.json (~632 entries; ~527
    appear in eval). Return (label_list, normalized_name -> index)."""
    with open(ONTOLOGY_JSON) as f:
        ont = json.load(f)
    labels = [e["name"] for e in ont]
    name_to_idx = {_norm(n): i for i, n in enumerate(labels)}
    return labels, name_to_idx


def load_eval(max_samples: int | None) -> list[dict]:
    """Read all AudioSet eval parquet files and return rows.

    Each row: {video_id, audio_bytes (FLAC), labels (mid IDs), human_labels (text)}.
    Pre-decoded waveform may be loaded later via _DECODED_CACHE env var.
    """
    import pyarrow.parquet as pq
    rows: list[dict] = []
    files = sorted(EVAL_PARQUET_DIR.glob("*.parquet"))
    for pf in files:
        try:
            table = pq.read_table(pf)
        except Exception as e:
            print(f"[audioset] read fail {pf.name}: {e}", flush=True)
            continue
        df = table.to_pandas()
        for _, r in df.iterrows():
            hv = r.get("human_labels")
            if hv is None:
                continue
            human = [str(x) for x in list(hv)]
            if not human:
                continue
            audio = r["audio"]
            # parquet `audio` is a dict {bytes, path}; we want bytes.
            wav_bytes = audio["bytes"] if isinstance(audio, dict) else audio
            rows.append({
                "fname": str(r["video_id"]),
                "audio_bytes": wav_bytes,
                "labels": human,  # use human-readable strings
            })
            if max_samples is not None and len(rows) >= max_samples:
                return rows
    return rows


def preprocess_audio_bytes(wav_bytes: bytes, target_sr: int, max_samples: int) -> torch.Tensor:
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
# inference helpers (mirrors eval_fsd50k_map.py)
# ---------------------------------------------------------------------------


def parse_labels(text: str, name_to_idx: dict[str, int]) -> list[int]:
    """Greedy mode: parse comma-separated label string -> indices into vocab."""
    out = []
    for chunk in text.replace(";", ",").split(","):
        n = _norm(chunk)
        if n in name_to_idx:
            out.append(name_to_idx[n])
    return list(set(out))


def parse_labels_sentence(text: str, vocab: list[str]) -> list[int]:
    """Sentence mode: substring-match each label name (and optional plural) in
    the free-form description. Word-boundary aware."""
    text_lower = text.lower()
    out = set()
    for idx, name in enumerate(vocab):
        # Multi-word labels: tolerate single space variations, optional plural at end.
        words = name.lower().split()
        # Build a regex like "\bword1\s+word2\s+word3s?\b"
        pat = r"\b" + r"\s+".join(re.escape(w) for w in words[:-1] + [words[-1]]) + r"s?\b"
        if re.search(pat, text_lower):
            out.add(idx)
    return sorted(out)


def run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache=True, stem=EVAL_STEM):
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in batch:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        ids = build_prompt_ids(tokenizer, audio_pad_id, t_audio, stem)
        prompts.append(ids)
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
    )


def evaluate_one(
    ckpt_path: str,
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

    # Optional: pre-decoded waveform cache (set AUDIOSET_DECODED_CACHE env var).
    # Cache is encoder-specific (DAC 48 kHz vs Whisper 16 kHz).
    cache_path = _os.environ.get("AUDIOSET_DECODED_CACHE")
    cache: dict[str, torch.Tensor] | None = None
    if cache_path and Path(cache_path).exists():
        print(f"[audioset] loading pre-decoded cache <- {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu")
        print(f"[audioset] cache: {len(cache)} entries", flush=True)

    print(f"[audioset] decoding {len(rows)} audios at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows:
        try:
            if cache is not None:
                wav_fp16 = cache.get(r["fname"])
                if wav_fp16 is None:
                    wav = preprocess_audio_bytes(r["audio_bytes"], target_sr, max_audio_samples)
                else:
                    wav = wav_fp16.to(torch.float32)
            else:
                wav = preprocess_audio_bytes(r["audio_bytes"], target_sr, max_audio_samples)
        except Exception as e:
            print(f"[audioset] load fail {r['fname']}: {e}", flush=True)
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
        parse_fn = lambda txt: parse_labels(txt, vocab_norm)

    n_labels = len(vocab)
    y_true = np.zeros((len(prepared), n_labels), dtype=np.uint8)
    y_pred = np.zeros((len(prepared), n_labels), dtype=np.uint8)

    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        for bs_start in range(0, len(prepared), batch_size):
            batch = prepared[bs_start:bs_start + batch_size]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch, active_max_new, use_cache=use_cache, stem=active_stem)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyps = []
                for one in batch:
                    hyps.extend(run_batch(model, tokenizer, cfg, [one], active_max_new, use_cache=use_cache, stem=active_stem))
            for i, (r, hyp) in enumerate(zip(batch, hyps)):
                idx = bs_start + i
                # gold (human labels)
                gold_idxs = []
                for lab in r["labels"]:
                    n = _norm(lab)
                    if n in vocab_norm:
                        gold_idxs.append(vocab_norm[n])
                pred_idxs = parse_fn(hyp)
                y_true[idx, gold_idxs] = 1
                y_pred[idx, pred_idxs] = 1
                fp.write(json.dumps({
                    "fname": r["fname"],
                    "gold": list(r["labels"]),
                    "pred_text": hyp,
                    "pred_labels": [vocab[j] for j in pred_idxs],
                }, ensure_ascii=False) + "\n")
            if (bs_start // batch_size) % 20 == 0:
                el = time.time() - t0
                done = bs_start + len(batch)
                print(f"[audioset] {done}/{len(prepared)}  ({done/el:.2f} sps)", flush=True)

    # F1-micro/macro + Jaccard
    tp = (y_true & y_pred).sum(axis=0)
    fp_v = ((1 - y_true) & y_pred).sum(axis=0)
    fn_v = (y_true & (1 - y_pred)).sum(axis=0)
    prec_macro = np.where((tp + fp_v) > 0, tp / np.maximum(tp + fp_v, 1), 0.0)
    rec_macro = np.where((tp + fn_v) > 0, tp / np.maximum(tp + fn_v, 1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_per = np.where((prec_macro + rec_macro) > 0,
                          2 * prec_macro * rec_macro / (prec_macro + rec_macro), 0.0)
    # only labels with positive support count toward macro
    has_support = (y_true.sum(axis=0) > 0)
    f1_macro = float(f1_per[has_support].mean()) if has_support.any() else 0.0
    tp_s, fp_s, fn_s = int(tp.sum()), int(fp_v.sum()), int(fn_v.sum())
    prec_micro = tp_s / max(tp_s + fp_s, 1)
    rec_micro = tp_s / max(tp_s + fn_s, 1)
    f1_micro = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0

    # Jaccard (per-sample average IoU over labels)
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    jacc = float(np.where(union > 0, inter / np.maximum(union, 1), 0.0).mean())

    summary = {
        "checkpoint": str(ckpt_path),
        "n": len(prepared),
        "n_labels": n_labels,
        "n_labels_with_support": int(has_support.sum()),
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "jaccard": jacc,
        "elapsed_sec": time.time() - t0,
        "stem": active_stem,
        "score_mode": score_mode,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return summary


def _run_sequence_scoring(model, tokenizer, cfg, prepared, vocab, vocab_norm,
                          out_dir, ckpt_path, label_batch_size: int = 50):
    """mAP via teacher-forced per-label log-prob (mirrors fsd50k seq mode)."""
    from sklearn.metrics import average_precision_score
    n_labels = len(vocab)
    # Pre-tokenize each vocab label string -> token id list (raw, no leading space).
    label_token_seqs = [tokenizer.encode(v, add_special_tokens=False) for v in vocab]
    y_true = np.zeros((len(prepared), n_labels), dtype=np.uint8)
    y_score = np.zeros((len(prepared), n_labels), dtype=np.float32)
    t0 = time.time()
    pred_path = out_dir / "predictions.jsonl"
    audio_pad_id = cfg.audio_pad_token_id
    with open(pred_path, "w", encoding="utf-8") as fp:
        for i, r in enumerate(prepared):
            wav = r["_wav"]
            t_audio = t_audio_for(cfg, wav.shape[-1])
            prompt_ids = build_prompt_ids(tokenizer, audio_pad_id, t_audio, EVAL_STEM)
            try:
                scores = score_labels_teacher_forced(
                    model, cfg, prompt_ids, wav, label_token_seqs,
                    batch_size=label_batch_size, length_normalize=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                scores = score_labels_teacher_forced(
                    model, cfg, prompt_ids, wav, label_token_seqs,
                    batch_size=max(1, label_batch_size // 4), length_normalize=True,
                )
            scores = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
            scores = scores.reshape(-1)
            y_score[i, :] = scores
            for lab in r["labels"]:
                n = _norm(lab)
                if n in vocab_norm:
                    y_true[i, vocab_norm[n]] = 1
            fp.write(json.dumps({
                "fname": r["fname"],
                "gold": list(r["labels"]),
                "top10_pred": [vocab[j] for j in np.argsort(-scores)[:10]],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"[audioset-seq] {i+1}/{len(prepared)}  ({(i+1)/el:.3f} sps)", flush=True)

    has_support = (y_true.sum(axis=0) > 0)
    if has_support.any():
        ap_macro = average_precision_score(y_true[:, has_support], y_score[:, has_support],
                                            average="macro")
        ap_micro = average_precision_score(y_true[:, has_support], y_score[:, has_support],
                                            average="micro")
    else:
        ap_macro = ap_micro = 0.0
    summary = {
        "checkpoint": str(ckpt_path),
        "n": len(prepared),
        "n_labels": n_labels,
        "n_labels_with_support": int(has_support.sum()),
        "mAP_macro": float(ap_macro),
        "mAP_micro": float(ap_micro),
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_STEM,
        "score_mode": "sequence",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None,
                   help="Base model dir for adapter-only ckpts (default: read from adapter_config.json)")
    p.add_argument("--ckpts", default="all")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap rows (debug/smoke). default=all 17k.")
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--score-mode", choices=["greedy", "sequence", "sentence"], default="greedy",
                   help="greedy = canonical comma-list prompt; sentence = free-form description "
                        "with substring label matching; sequence = per-label teacher-forced scoring (mAP).")
    p.add_argument("--label-batch-size", type=int, default=50,
                   help="sequence-mode: labels per teacher-force forward")
    args = p.parse_args()

    vocab, vocab_norm = load_vocab()
    print(f"[audioset] vocab: {len(vocab)} labels", flush=True)
    rows = load_eval(args.max_samples)
    print(f"[audioset] eval rows: {len(rows)}", flush=True)

    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    steps_filter = None
    if args.ckpts and args.ckpts != "all":
        steps_filter = {int(s) for s in args.ckpts.split(",") if s.strip()}
    ckpts = find_checkpoints(ckpt_root, steps_filter)
    print(f"[audioset] ckpts: {[c.name for c in ckpts]}", flush=True)

    all_results = {}
    for ck in ckpts:
        out_dir = out_root / ck.name
        if (out_dir / "summary.json").exists():
            print(f"[audioset] skip {ck.name}: summary exists")
            with open(out_dir / "summary.json") as f:
                all_results[ck.name] = json.load(f)
            continue
        s = evaluate_one(
            ckpt_path=str(ck),
            base_model=args.base_model,
            rows=rows,
            vocab=vocab,
            vocab_norm=vocab_norm,
            batch_size=args.batch_size,
            out_dir=out_dir,
            max_audio_samples=args.max_audio_samples,
            max_new_tokens=args.max_new_tokens,
            use_cache=not args.no_cache,
            score_mode=args.score_mode,
            label_batch_size=args.label_batch_size,
        )
        all_results[ck.name] = s

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY ===")
    if args.score_mode == "sequence":
        print(f"{'ckpt':30s}  {'mAP-mi':>6s}  {'mAP-ma':>6s}")
        for name, s in all_results.items():
            print(f"{name:30s}  {s['mAP_micro']:6.4f}  {s['mAP_macro']:6.4f}")
    else:
        # greedy + sentence both report f1_micro / f1_macro / jaccard
        print(f"{'ckpt':30s}  {'F1-mi':>6s}  {'F1-ma':>6s}  {'Jacc':>6s}")
        for name, s in all_results.items():
            print(f"{name:30s}  {s['f1_micro']:6.4f}  {s['f1_macro']:6.4f}  {s['jaccard']:6.4f}")


if __name__ == "__main__":
    main()
