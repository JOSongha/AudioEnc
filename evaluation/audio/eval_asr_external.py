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


# Tag → nubes loader spec. Bypasses HF; uses nubes gateway via _nubes_loader.
# librispeech_*: transcripts as jsonl + flat wav dir (same layout as eval_librispeech_wer.py).
# gigaspeech: paired `<id>.flac` + `<id>.txt` under one dir (2024-01-04 외부 팀 본).
_NUBES_LOADERS: dict[str, dict] = {
    "librispeech_clean": {
        "type": "jsonl_transcript",
        "transcript": "users/jos/AudioEnc/LibriSpeech/test_clean.jsonl",
        "audio_prefix": "datasets/public/librispeech_asr/clean/test/",
        "audio_ext": ".wav",
    },
    "librispeech_other": {
        "type": "jsonl_transcript",
        "transcript": "users/jos/AudioEnc/LibriSpeech/test_other.jsonl",
        "audio_prefix": "datasets/public/librispeech_asr/other/test/",
        "audio_ext": ".wav",
    },
    "gigaspeech": {
        "type": "paired_files",
        "audio_prefix": "datasets/public/16kHz/gigaspeech/test/",
        "audio_ext": ".flac",
        "text_ext": ".txt",
    },
}


def _load_nubes_jsonl_transcript(spec: dict, tag: str, max_samples: int | None) -> list[dict]:
    """jsonl 라인-별 `{id,text}` + `<audio_prefix>/<id><ext>` audio. LibriSpeech 패턴."""
    from evaluation.audio._nubes_loader import fetch_nubes_text, fetch_nubes_audio_tensor
    transcript = fetch_nubes_text(spec["transcript"])
    pairs = [json.loads(l) for l in transcript.splitlines() if l.strip()]
    if max_samples is not None:
        pairs = pairs[:max_samples]
    rows = []
    for r in pairs:
        try:
            wav, sr = fetch_nubes_audio_tensor(
                f"{spec['audio_prefix']}{r['id']}{spec['audio_ext']}")
        except Exception as e:
            print(f"[asr-ext] {tag} audio fetch fail {r['id']}: {e}", flush=True)
            continue
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        rows.append({"id": r["id"], "text": r["text"],
                     "_wav": wav.squeeze(0).to(torch.float32), "_sr": int(sr)})
    return rows


def _load_nubes_paired_files(spec: dict, tag: str, max_samples: int | None) -> list[dict]:
    """`<prefix>/<id>.{audio,text}` 페어. GigaSpeech 패턴."""
    from evaluation.audio._nubes_loader import (
        list_nubes_dir, fetch_nubes_text, fetch_nubes_audio_tensor)
    rows = []
    audio_ext = spec["audio_ext"]
    text_ext = spec["text_ext"]
    for audio_path in list_nubes_dir(spec["audio_prefix"], suffix=audio_ext):
        if max_samples is not None and len(rows) >= max_samples:
            break
        stem = audio_path.rsplit("/", 1)[-1].removesuffix(audio_ext)
        text_path = f"{spec['audio_prefix']}{stem}{text_ext}"
        try:
            text = fetch_nubes_text(text_path).strip()
            wav, sr = fetch_nubes_audio_tensor(audio_path)
        except Exception as e:
            print(f"[asr-ext] {tag} pair fetch fail {stem}: {e}", flush=True)
            continue
        if not text:
            continue
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        rows.append({"id": stem, "text": text,
                     "_wav": wav.squeeze(0).to(torch.float32), "_sr": int(sr)})
    return rows


def _load_nubes(tag: str, max_samples: int | None) -> list[dict]:
    spec = _NUBES_LOADERS[tag]
    print(f"[asr-ext] loading {tag} <- nubes ({spec['type']})", flush=True)
    if spec["type"] == "jsonl_transcript":
        rows = _load_nubes_jsonl_transcript(spec, tag, max_samples)
    elif spec["type"] == "paired_files":
        rows = _load_nubes_paired_files(spec, tag, max_samples)
    else:
        raise ValueError(f"unknown nubes loader type: {spec['type']!r}")
    print(f"[asr-ext] {tag} loaded n={len(rows)}", flush=True)
    return rows


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
    # nubes-direct route: librispeech_{clean,other} + gigaspeech
    if tag in _NUBES_LOADERS:
        return _load_nubes(tag, max_samples)
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


def _eval_one_ckpt(ckpt_path: Path, base_model: str | None,
                   datasets_list: list[str], max_samples: int,
                   batch_size: int, max_new_tokens: int, use_cache: bool,
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    summary = {"checkpoint": str(ckpt_path), "max_samples": max_samples, "datasets": {}}
    for tag in datasets_list:
        try:
            rows = load_split(tag, max_samples)
        except Exception as e:
            print(f"[asr-ext] {tag} FAILED to load: {e}", flush=True)
            summary["datasets"][tag] = {"status": "load_failed", "error": str(e)}
            continue
        if not rows:
            summary["datasets"][tag] = {"status": "no_rows"}
            continue
        try:
            r = run_dataset(model, tokenizer, cfg, rows,
                            batch_size=batch_size,
                            max_new_tokens=max_new_tokens,
                            use_cache=use_cache)
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True,
                   help="Parent dir containing checkpoint-* subdirs, OR a single "
                        "checkpoint dir (legacy). With --ckpts, --ckpt-root is parent.")
    p.add_argument("--ckpts", default=None,
                   help="Comma-separated step filter. Loops over matching ckpts.")
    p.add_argument("--base-model", default=None)
    p.add_argument("--out-root", default=None,
                   help="default = <ckpt-root>/eval_asr_external")
    p.add_argument("--datasets", nargs="+", default=["mls", "voxpopuli", "gigaspeech"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--include-partial", action="store_true")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root) if args.out_root else ckpt_root / "eval_asr_external"
    out_root.mkdir(parents=True, exist_ok=True)

    # Two modes:
    # 1) --ckpts given: ckpt_root is parent, find_checkpoints filters by step.
    # 2) --ckpts absent: ckpt_root is a single ckpt path (legacy single-ckpt mode).
    if args.ckpts is not None:
        steps_filter = {int(s) for s in args.ckpts.split(",") if s.strip()}
        ckpts = find_checkpoints(ckpt_root, steps_filter,
                                 min_age_sec=0 if args.include_partial else 60)
        if not ckpts:
            raise SystemExit(f"[asr-ext] no checkpoints under {ckpt_root} matching {steps_filter}")
    else:
        ckpts = [ckpt_root]

    all_summaries = []
    for p_ckpt in ckpts:
        ckpt_out = out_root / p_ckpt.name if args.ckpts is not None else out_root
        summary = _eval_one_ckpt(
            p_ckpt, args.base_model, args.datasets, args.max_samples,
            args.batch_size, args.max_new_tokens, not args.no_cache, ckpt_out,
        )
        all_summaries.append(summary)
        print(f"\n[asr-ext] wrote {ckpt_out / 'summary.json'}", flush=True)
        print(f"\n=== SUMMARY ({p_ckpt.name}) ===")
        print(f"{'dataset':<20} {'n':>6} {'WER%':>7} {'CER%':>7} {'note'}")
        for tag, r in summary["datasets"].items():
            if r.get("status") == "ok":
                print(f"{tag:<20} {r['n']:>6} {r['wer_normalized']*100:>7.2f} "
                      f"{r['cer_normalized']*100:>7.2f}  {r.get('license_note','')}")
            else:
                print(f"{tag:<20} {'-':>6} {'-':>7} {'-':>7}  {r.get('status', '?')}")
    if len(all_summaries) > 1:
        (out_root / "summary_all.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
