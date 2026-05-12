"""AudioCaps test caption eval for Qwen3.5AE Stage-2 checkpoints.

Mirrors [`eval_clotho_caption.py`](eval_clotho_caption.py) but reads HuggingFace
AudioCaps test parquet (`OpenSound/AudioCaps`, distributed by jos to nubes
`/users/jos/AudioEnc/AudioCaps/data/test-*.parquet`, § 12.9). Each parquet row
= 1 caption + embedded audio bytes; 5 captions per (youtube_id, start_time)
unique audio. We group rows by audio key → list of 5 captions → caption eval.

Held-out integrity:
- v6 training pool only uses train (412 parquet) + validation (20 parquet),
  see `scripts/manifest_builders/build_audiocaps.py:SPLIT_PARQUET_COUNT`. The
  test 41 parquet (883 unique audio, 4,411 row) is **not enumerated by builder**
  (calling `list_split_parquets("test")` raises `ValueError`). So this eval is
  leak-free against any v6 Stage-1 ckpt.
- Stage-2 LISTEN-mix contamination caveat does NOT apply here (LISTEN doesn't
  pull AudioCaps test rows — only MELD-test and MOSEI-test, see
  `eval_source_emotion.py:11-18` docstring).

Usage:
    python -m evaluation.audio.eval_audiocaps_caption \\
        --ckpt-root .../results/Qwen3.5AE-Stage2-... \\
        --out-root  .../eval_audiocaps \\
        --base-model /mnt/tmp/s2_init_42k \\
        --ckpts 12000 --batch-size 4
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchaudio

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

EVAL_CAPTION_STEM = "Describe what you hear in the audio."
MAX_NEW_TOKENS = 96


# ---------------------------------------------------------------------------
# BLEU-N (mirrors eval_clotho_caption.corpus_bleu — identical implementation)
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(s: str) -> list[str]:
    return [w.lower() for w in _TOK_RE.findall(s)]


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(list_of_refs: list[list[list[str]]], hyps: list[list[str]],
                max_n: int = 4) -> tuple[float, list[float]]:
    assert len(list_of_refs) == len(hyps)
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    hyp_len = 0
    for refs, hyp in zip(list_of_refs, hyps):
        hyp_len += len(hyp)
        ref_len += min(
            (abs(len(r) - len(hyp)), len(r)) for r in refs
        )[1] if refs else len(hyp)
        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            max_ref_ng: Counter = Counter()
            for r in refs:
                r_ng = _ngrams(r, n)
                for k, v in r_ng.items():
                    if v > max_ref_ng[k]:
                        max_ref_ng[k] = v
            for ng, c in hyp_ng.items():
                clipped[n - 1] += min(c, max_ref_ng[ng])
            totals[n - 1] += sum(hyp_ng.values())
    precisions = [(clipped[n] / totals[n]) if totals[n] else 0.0 for n in range(max_n)]
    geo = 0.0 if min(precisions) == 0 else math.exp(
        sum(math.log(p) for p in precisions) / max_n)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    return bp * geo, precisions


# ---------------------------------------------------------------------------
# data loading — stream test parquets from nubes
# ---------------------------------------------------------------------------

NUBES_TEST_PARQUET_DIR = "users/jos/AudioEnc/AudioCaps/data/"


def load_audiocaps_test() -> list[dict]:
    """Return list of {file_key, audio_bytes, captions[5]}.

    Streams all `test-*.parquet` (41 file, ~4,411 row, 883 unique audio) from
    nubes, decodes parquet bytes in-memory, groups rows by (youtube_id,
    start_time) so each entry has all 5 captions. Audio bytes are taken from
    the FIRST row per group (identical across the 5 caption rows).
    """
    from evaluation.audio._nubes_loader import (
        list_nubes_dir, fetch_nubes_bytes)
    import pyarrow.parquet as pq

    grouped: dict[str, dict] = {}
    n_parquet = 0
    for pf in list_nubes_dir(NUBES_TEST_PARQUET_DIR, suffix=".parquet"):
        if not pf.rsplit("/", 1)[-1].startswith("test-"):
            continue
        n_parquet += 1
        body = fetch_nubes_bytes(pf)
        table = pq.read_table(io.BytesIO(body))
        df = table.to_pandas()
        for _, r in df.iterrows():
            key = f"{r['youtube_id']}_{int(r['start_time'])}"
            cap = str(r["caption"]).strip()
            if not cap:
                continue
            if key not in grouped:
                audio = r["audio"]
                wav_bytes = audio["bytes"] if isinstance(audio, dict) else audio
                grouped[key] = {
                    "file": key,
                    "audio_bytes": wav_bytes,
                    "captions": [cap],
                }
            else:
                grouped[key]["captions"].append(cap)
    print(f"[audiocaps] streamed {n_parquet} test parquets → "
          f"{len(grouped)} unique audio (avg "
          f"{sum(len(v['captions']) for v in grouped.values()) / max(len(grouped), 1):.1f} "
          f"caps/audio)", flush=True)
    return list(grouped.values())


def preprocess_audio_bytes(wav_bytes: bytes, target_sr: int,
                           max_samples: int) -> torch.Tensor:
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
# generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_batch(model, tokenizer, cfg, batch: list[dict],
              max_new_tokens: int, use_cache: bool = True) -> list[str]:
    audio_pad_id = cfg.audio_pad_token_id
    prompts, waveforms = [], []
    for r in batch:
        wav = r["_wav"]
        t_audio = t_audio_for(cfg, wav.shape[-1])
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio, EVAL_CAPTION_STEM))
        waveforms.append(wav)
    return generate_greedy(
        model, tokenizer, cfg, prompts, waveforms,
        max_new_tokens=max_new_tokens, use_cache=use_cache,
    )


def eval_checkpoint(
    ckpt_path: Path,
    base_model: str | None,
    rows: list[dict],
    batch_size: int,
    out_dir: Path,
    max_audio_samples: int | None,
    max_new_tokens: int,
    try_pycoco: bool = True,
    use_cache: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    model, tokenizer, cfg = load_checkpoint(ckpt_path, base_model_dir=base_model)
    target_sr = audio_sample_rate(cfg)
    if max_audio_samples is None:
        max_audio_samples = default_max_audio_samples(cfg)

    print(f"[audiocaps] decoding {len(rows)} audios at {target_sr} Hz...", flush=True)
    prepared = []
    for r in rows:
        try:
            wav = preprocess_audio_bytes(r["audio_bytes"], target_sr, max_audio_samples)
        except Exception as e:
            print(f"[audiocaps] decode fail {r['file']}: {e}", flush=True)
            continue
        prepared.append({**r, "_wav": wav})
    prepared.sort(key=lambda x: x["_wav"].shape[-1])

    all_preds: list[str] = []
    all_refs: list[list[list[str]]] = []

    t0 = time.time()
    with open(pred_path, "w", encoding="utf-8") as fp:
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch, max_new_tokens, use_cache=use_cache)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyps = []
                for one in batch:
                    hyps.extend(run_batch(model, tokenizer, cfg, [one], max_new_tokens, use_cache=use_cache))

            for r, hyp in zip(batch, hyps):
                all_preds.append(hyp)
                all_refs.append([tokenize(c) for c in r["captions"]])
                fp.write(json.dumps({
                    "file": r["file"],
                    "hyp": hyp,
                    "refs": r["captions"],
                }, ensure_ascii=False) + "\n")

            done = i + len(batch)
            if (i // batch_size) % 10 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                print(f"[audiocaps] {ckpt_path.name} {done}/{len(prepared)}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f}m", flush=True)

    hyp_toks = [tokenize(h) for h in all_preds]
    bleu4, precisions = corpus_bleu(all_refs, hyp_toks, max_n=4)
    bleu1 = precisions[0] if precisions else 0.0

    coco_scores: dict[str, float | None] = {"CIDEr": None, "METEOR": None,
                                             "ROUGE_L": None, "SPICE": None}
    if try_pycoco:
        gts = {str(i): r["captions"] for i, r in enumerate(prepared)}
        res = {str(i): [all_preds[i]] for i in range(len(all_preds))}
        try:
            from pycocoevalcap.cider.cider import Cider
            coco_scores["CIDEr"] = float(Cider().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[audiocaps] CIDEr unavailable: {e}", flush=True)
        try:
            from pycocoevalcap.rouge.rouge import Rouge
            coco_scores["ROUGE_L"] = float(Rouge().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[audiocaps] ROUGE_L unavailable: {e}", flush=True)
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            coco_scores["METEOR"] = float(Meteor().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[audiocaps] METEOR unavailable (needs Java): {e}", flush=True)
        try:
            from pycocoevalcap.spice.spice import Spice
            coco_scores["SPICE"] = float(Spice().compute_score(gts, res)[0])
        except Exception as e:
            print(f"[audiocaps] SPICE unavailable (needs Java + CoreNLP): {e}", flush=True)

    summary = {
        "checkpoint": str(ckpt_path),
        "n": len(prepared),
        "bleu1": bleu1,
        "bleu4": bleu4,
        "precisions_1to4": precisions,
        **coco_scores,
        "elapsed_sec": time.time() - t0,
        "stem": EVAL_CAPTION_STEM,
        "max_audio_samples": max_audio_samples,
        "use_cache": use_cache,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[audiocaps] {ckpt_path.name} BLEU-1={bleu1:.4f} BLEU-4={bleu4:.4f} "
          f"CIDEr={coco_scores['CIDEr']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--ckpts", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-audio-samples", type=int, default=None,
                   help="Default: 1.6M (DAC) / 480k (Whisper) — chosen from cfg.")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-pycoco", action="store_true",
                   help="Skip the CIDEr/METEOR/ROUGE block even if pycocoevalcap is installed.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable KV/conv cache during generation.")
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
        raise SystemExit(f"[audiocaps] no checkpoints under {ckpt_root}")

    rows = load_audiocaps_test()
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[audiocaps] rows={len(rows)}", flush=True)

    all_summaries = []
    for p in ckpts:
        out_dir = out_root / p.name
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue
        try:
            s = eval_checkpoint(
                p, args.base_model, rows,
                batch_size=args.batch_size,
                out_dir=out_dir,
                max_audio_samples=args.max_audio_samples,
                max_new_tokens=args.max_new_tokens,
                try_pycoco=not args.no_pycoco,
                use_cache=not args.no_cache,
            )
            all_summaries.append(s)
        except Exception:
            import traceback
            traceback.print_exc()

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'ckpt':<24s}  {'BLEU-1':>7s}  {'BLEU-4':>7s}  {'CIDEr':>7s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        c = s.get("CIDEr")
        c_str = f"{c:.4f}" if isinstance(c, (int, float)) else "   -   "
        print(f"{name:<24s}  {s['bleu1']:>7.4f}  {s['bleu4']:>7.4f}  {c_str:>7s}")


if __name__ == "__main__":
    main()
