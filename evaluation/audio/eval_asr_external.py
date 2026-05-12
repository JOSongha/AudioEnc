"""ASR WER on external datasets (OOD overfit check).

Mirrors `eval_librispeech_wer.py` flow but supports multiple HF ASR datasets.
Tag interpretation depends on which training stage's ckpt is being evaluated:

                       Stage-1 v6 mix         Stage-2 LoRA mix
  MLS (en)             in-dist                in-dist
  VoxPopuli (en)       in-dist                in-dist
  LibriSpeech test     in-dist (LibriTTS-R    in-dist (LibriTTS-R
                       train.* shares text/   train.* shares text/
                       speakers but disjoint  speakers but disjoint
                       splits)                splits)
  GigaSpeech           in-dist (XL train      in-dist
                       50% sample, disjoint
                       test split)
  CommonVoice          OOD                    in-dist

Stage-1 v6 ASR pool = {MLS, LibriTTS-R train.*, VoxPopuli train, GigaSpeech XL
50% random subsample (seed=11)}. GigaSpeech sampling shares the train pool with
the held-out test split, so eval_asr_external --datasets gigaspeech is still a
valid leak-free WER probe. CommonVoice is pulled in only at Stage-2 (LISTEN/
external mix), so it remains Stage-1 OOD. See docs/setup/datasets.md §2.

Subsamples to MAX_SAMPLES (default 500) for tractable cross-dataset comparison.

Usage:
    python -m evaluation.audio.eval_asr_external \\
        --ckpt-root /mnt/tmp/results/.../checkpoint-15000 \\
        --base-model /mnt/tmp/s2_init_42k \\
        --datasets mls voxpopuli gigaspeech \\
        --max-samples 500 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.audio._loader import (  # noqa: E402
    audio_sample_rate,
    build_prompt_ids,
    default_max_audio_samples,
    t_audio_for,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
)
from evaluation.audio.eval_librispeech_wer import ASR_STEM  # noqa: E402

import jiwer  # noqa: E402
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer  # noqa: E402

_NORMALIZER = EnglishTextNormalizer({})


def normalize_for_wer(s: str) -> str:
    return _NORMALIZER(s).strip()


def compute_wer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def compute_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    return jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])

# Dataset registry: tag → (HF repo, config, split, audio_key, text_key, license_note)
# license_note reads "<Stage-1 v6 status> | <Stage-2 status>"
DATASETS = {
    "librispeech_clean": (
        "openslr/librispeech_asr", "clean", "test", "audio", "text",
        "Stage-1 in-dist (LibriTTS-R train.* shares speakers/text, disjoint test split) | Stage-2 in-dist"),
    "librispeech_other": (
        "openslr/librispeech_asr", "other", "test", "audio", "text",
        "Stage-1 in-dist (LibriTTS-R train.* shares speakers/text, disjoint test split) | Stage-2 in-dist"),
    "mls": (
        "parler-tts/mls_eng_10k", None, "test", "audio", "transcript",
        "Stage-1 in-dist (training pool, train split) | Stage-2 in-dist"),
    "voxpopuli": (
        "facebook/voxpopuli", "en", "test", "audio", "raw_text",
        "Stage-1 in-dist (training pool, train split) | Stage-2 in-dist"),
    "gigaspeech": (
        "speechcolab/gigaspeech", "test", "test", "audio", "text",
        "Stage-1 in-dist (XL train 50% sample, disjoint test split) | Stage-2 in-dist"),
    "commonvoice": (
        "mozilla-foundation/common_voice_17_0", "en", "test", "audio", "sentence",
        "Stage-1 OOD (not in v6 ASR pool) | Stage-2 in-dist (HF gate)"),
    "commonvoice_local": (
        "/mnt/ddn/omni_dataset/audio/common_voice/common_voice_test.jsonl", None, None, None, None,
        "Stage-1 OOD (not in v6 ASR pool) | Stage-2 in-dist (local jsonl, bypasses HF gate)"),
}


def _load_local_jsonl(path: str, max_samples: int | None) -> list[dict]:
    """Load a {text, audio} jsonl with absolute audio paths."""
    import torchaudio
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            txt = r.get("text") or r.get("transcript") or r.get("sentence")
            ap = r.get("audio") or r.get("path") or r.get("file")
            if not txt or not ap or not Path(ap).exists():
                continue
            try:
                wav, sr = torchaudio.load(ap)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0).to(torch.float32)
            except Exception:
                continue
            rows.append({
                "id": Path(ap).stem,
                "text": txt,
                "_wav": wav,
                "_sr": int(sr),
            })
    return rows


def load_split(tag: str, max_samples: int | None) -> list[dict]:
    repo, config, split, audio_key, text_key, _ = DATASETS[tag]
    # Local jsonl path (e.g. commonvoice_local)
    if repo.startswith("/") and Path(repo).is_file():
        print(f"[asr-ext] loading {tag} <- {repo} (local jsonl)", flush=True)
        rows = _load_local_jsonl(repo, max_samples)
        print(f"[asr-ext] {tag} loaded n={len(rows)}", flush=True)
        return rows
    from datasets import load_dataset
    kwargs = {"split": split, "cache_dir": "/mnt/tmp/cache"}
    if config is not None:
        kwargs["name"] = config
    print(f"[asr-ext] loading {tag} <- {repo} (config={config}, split={split})", flush=True)
    ds = load_dataset(repo, **kwargs, trust_remote_code=True)
    rows = []
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(n):
        r = ds[i]
        a = r[audio_key]
        wav = torch.tensor(a["array"], dtype=torch.float32)
        sr = int(a["sampling_rate"])
        text = r[text_key]
        if not text or not isinstance(text, str):
            continue
        rows.append({
            "id": r.get("id") or r.get("audio_id") or f"{tag}_{i}",
            "text": text,
            "_wav": wav,
            "_sr": sr,
        })
    print(f"[asr-ext] {tag} loaded n={len(rows)}", flush=True)
    return rows


@torch.inference_mode()
def run_dataset(model, tokenizer, cfg, rows, *, batch_size, max_new_tokens, use_cache):
    from torchaudio.functional import resample
    target_sr = audio_sample_rate(cfg)
    audio_pad_id = cfg.audio_pad_token_id

    preds, refs = [], []
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts, waveforms = [], []
        for r in batch:
            wav = r["_wav"]
            if r["_sr"] != target_sr:
                wav = resample(wav, r["_sr"], target_sr)
            t_audio = t_audio_for(cfg, wav.shape[-1])
            prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, ASR_STEM))
            waveforms.append(wav)
        outputs = generate_greedy(
            model, tokenizer, cfg, prompts, waveforms,
            max_new_tokens=max_new_tokens, use_cache=use_cache,
        )
        for r, o in zip(batch, outputs):
            preds.append(o)
            refs.append(r["text"])
    elapsed = time.time() - t0

    norm_pairs = [(normalize_for_wer(r), normalize_for_wer(p)) for r, p in zip(refs, preds)]
    norm_refs = [pair[0] for pair in norm_pairs]
    norm_preds = [pair[1] for pair in norm_pairs]
    wer_n = compute_wer(norm_refs, norm_preds)
    cer_n = compute_cer(norm_refs, norm_preds)
    return {
        "n": len(rows),
        "wer_normalized": float(wer_n),
        "cer_normalized": float(cer_n),
        "elapsed_sec": float(elapsed),
        "preds_sample": list(zip(refs[:5], preds[:5])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--out-root", default=None,
                   help="default = <ckpt-root>/eval_asr_external")
    p.add_argument("--datasets", nargs="+", default=["mls", "voxpopuli", "gigaspeech"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root) if args.out_root else ckpt_root / "eval_asr_external"
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, cfg = load_checkpoint(ckpt_root, base_model_dir=args.base_model)

    summary = {"checkpoint": str(ckpt_root), "max_samples": args.max_samples, "datasets": {}}
    for tag in args.datasets:
        try:
            rows = load_split(tag, args.max_samples)
        except Exception as e:
            print(f"[asr-ext] {tag} FAILED to load: {e}", flush=True)
            summary["datasets"][tag] = {"status": "load_failed", "error": str(e)}
            continue
        if not rows:
            summary["datasets"][tag] = {"status": "no_rows"}
            continue
        try:
            r = run_dataset(model, tokenizer, cfg, rows,
                            batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=not args.no_cache)
            r["status"] = "ok"
            r["license_note"] = DATASETS[tag][5]
            summary["datasets"][tag] = r
            print(f"[asr-ext] {tag}: WER={r['wer_normalized']*100:.2f}%  "
                  f"CER={r['cer_normalized']*100:.2f}%  n={r['n']}", flush=True)
        except Exception as e:
            print(f"[asr-ext] {tag} eval failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            summary["datasets"][tag] = {"status": "eval_failed", "error": str(e)}

    out_file = out_root / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[asr-ext] wrote {out_file}", flush=True)
    print("\n=== SUMMARY ===")
    print(f"{'dataset':<20} {'n':>6} {'WER%':>7} {'CER%':>7} {'note'}")
    for tag, r in summary["datasets"].items():
        if r.get("status") == "ok":
            print(f"{tag:<20} {r['n']:>6} {r['wer_normalized']*100:>7.2f} "
                  f"{r['cer_normalized']*100:>7.2f}  {r.get('license_note','')}")
        else:
            print(f"{tag:<20} {'-':>6} {'-':>7} {'-':>7}  {r.get('status', '?')}")


if __name__ == "__main__":
    main()
