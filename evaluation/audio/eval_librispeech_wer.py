"""LibriSpeech test-clean/test-other WER eval for Qwen3.5AE Stage-2 LoRA ckpts.

Uses our `_loader` (PEFT + projector + cache shims), so it works on
adapter-only checkpoints that the legacy `evaluation/eval_testclean_wer.py`
can't load (that script hardcodes AutoModel.from_pretrained on the ckpt dir,
which has no model.safetensors for PEFT saves).

Prompt matches training: canonical ASR stem "Transcribe the audio to text."
pinned to TASK_PROMPTS["asr"][0] and identical to eval_testclean_wer.py's
CHATML_MID. WER via Whisper's EnglishTextNormalizer (industry standard).

Usage:
    python -m evaluation.audio.eval_librispeech_wer \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17 \
        --out-root  .../eval_libri \
        --base-model /mnt/tmp/s2_init_42k \
        --split test.clean --ckpts 2000,12000 --batch-size 4
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import jiwer
import torch
import torchaudio
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.audio._loader import (  # noqa: E402
    audio_sample_rate,
    build_prompt_ids,
    default_max_audio_samples,
    find_checkpoints,
    generate_greedy,
    load_checkpoint,
    t_audio_for,
)

# Canonical ASR stem. Same string as TASK_PROMPTS["asr"][0] and
# evaluation/eval_testclean_wer.py CHATML_MID's user-suffix.
ASR_STEM = "Transcribe the audio to text."
MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# data loading — use HF datasets (caches into /mnt/tmp/cache by default)
# ---------------------------------------------------------------------------


def load_split(split: str, max_samples: int | None) -> list[dict]:
    """Load LibriSpeech test-clean or test-other via HF datasets.

    Returns rows with `_wav` at the *native* sample rate of the dataset and the
    accompanying `_sr` field. Per-checkpoint resampling to the encoder's
    target SR happens in `eval_checkpoint` so the same `rows` list serves both
    DAC (48 kHz) and Whisper (16 kHz) encoders without duplicate decode passes.
    """
    from datasets import load_dataset
    # split in {"test.clean", "test.other"}. HF config = "clean"/"other",
    # split_name = "test".
    config = "clean" if "clean" in split else "other"
    ds = load_dataset(
        "openslr/librispeech_asr", config, split="test",
        cache_dir="/mnt/tmp/cache",
    )
    rows = []
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(n):
        r = ds[i]
        # `audio` is {array, sampling_rate, path}; use array directly.
        audio = r["audio"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        rows.append({
            "id": r["id"], "text": r["text"],
            "_wav": wav, "_sr": int(audio["sampling_rate"]),
        })
    return rows


# ---------------------------------------------------------------------------
# per-ckpt eval
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache,
              no_repeat_ngram, no_think):
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in batch:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        prompts.append(build_prompt_ids(
            tokenizer, audio_pad_id, t_audio, ASR_STEM, no_think=no_think,
        ))
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
        no_repeat_ngram_size=no_repeat_ngram,
    )


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    rows: list[dict],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int,
    max_new_tokens: int,
    use_cache: bool,
    no_repeat_ngram: int,
    no_think: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    target_sr = audio_sample_rate(cfg)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)
    normalizer = EnglishTextNormalizer({})

    # Per-encoder resample (rows from load_split keep native SR) + length cap.
    for r in rows:
        if r.get("_sr") != target_sr:
            r["_wav"] = torchaudio.functional.resample(r["_wav"], r["_sr"], target_sr)
            r["_sr"] = target_sr
        if r["_wav"].shape[-1] > max_audio_samples:
            r["_wav"] = r["_wav"][:max_audio_samples]
    # Length-sort for padding efficiency
    rows_sorted = sorted(rows, key=lambda x: x["_wav"].shape[-1])

    hyps_by_id: dict[str, tuple[str, str]] = {}
    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        for i in range(0, len(rows_sorted), batch_size):
            batch = rows_sorted[i : i + batch_size]
            try:
                outs = run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache,
                                 no_repeat_ngram, no_think)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                outs = []
                for one in batch:
                    outs.extend(run_batch(model, tokenizer, cfg, [one], max_new_tokens, use_cache,
                                          no_repeat_ngram, no_think))

            for r, hyp in zip(batch, outs):
                fp.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": hyp},
                                    ensure_ascii=False) + "\n")
                hyps_by_id[r["id"]] = (r["text"], hyp)

            done = i + len(batch)
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(rows_sorted) - done) / max(rate, 1e-6)
                print(f"[asr] {ckpt_path.name} {done}/{len(rows_sorted)}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    refs_raw = [v[0] for v in hyps_by_id.values()]
    hyps_raw = [v[1] for v in hyps_by_id.values()]
    refs_norm = [normalizer(x).strip() for x in refs_raw]
    hyps_norm = [normalizer(x).strip() for x in hyps_raw]
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r]
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]
    wer_norm = jiwer.wer(refs_f, hyps_f) if refs_f else float("nan")
    cer_norm = jiwer.cer(refs_f, hyps_f) if refs_f else float("nan")
    wer_raw = jiwer.wer(refs_raw, hyps_raw) if refs_raw else float("nan")

    summary = {
        "checkpoint": str(ckpt_path),
        "n_samples": len(rows_sorted),
        "n_scored": len(pairs),
        "wer_normalized": wer_norm,
        "cer_normalized": cer_norm,
        "wer_raw": wer_raw,
        "elapsed_sec": time.time() - t0,
        "stem": ASR_STEM,
        "use_cache": use_cache,
        "no_repeat_ngram": no_repeat_ngram,
        "no_think": no_think,
        "normalizer": "whisper.EnglishTextNormalizer",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[asr] {ckpt_path.name}  WER(norm)={wer_norm:.4f} "
          f"CER(norm)={cer_norm:.4f}  n={len(pairs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--split", default="test.clean",
                   choices=["test.clean", "test.other"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--include-partial", action="store_true")
    p.add_argument("--no-repeat-ngram", type=int, default=0,
                   help="no_repeat_ngram_size for generate; 0 = off (default).")
    p.add_argument("--no-think", action="store_true",
                   help="Append '/no_think' to the system prompt (Qwen3 reasoning suppression).")
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
        raise SystemExit(f"[asr] no checkpoints under {ckpt_root}")

    print(f"[asr] loading {args.split}...", flush=True)
    rows = load_split(args.split, args.max_samples)
    print(f"[asr] {len(rows)} rows", flush=True)

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        if (out_dir / "summary.json").exists():
            with open(out_dir / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, rows,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                use_cache=not args.no_cache,
                no_repeat_ngram=args.no_repeat_ngram,
                no_think=args.no_think,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'ckpt':<16s}  {'WER(norm)':>10s}  {'CER(norm)':>10s}  {'WER(raw)':>10s}  {'n':>5s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        print(f"{name:<16s}  {s['wer_normalized']:>10.4f}  "
              f"{s['cer_normalized']:>10.4f}  {s['wer_raw']:>10.4f}  "
              f"{s['n_scored']:>5d}")


if __name__ == "__main__":
    main()
