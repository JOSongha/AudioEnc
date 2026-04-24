"""LibriSpeech test-clean WER sweep across Qwen3.5AE-ASR checkpoints.

Runs greedy decoding on every checkpoint-* directory inside --ckpt-root,
dumps per-sample hyp/ref JSONL + aggregate summary. Single GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_testclean_wer.py \
        --ckpt-root /mnt/tmp/results/Qwen3.5AE-ASR-Stage1-libri_mls_vox \
        --out-root  /mnt/tmp/results/Qwen3.5AE-ASR-Stage1-libri_mls_vox/eval_testclean \
        [--max-samples 2620] [--batch-size 4] [--ckpts 46000,45000]
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import jiwer
import pyarrow.ipc as ipc
import torch
import torchaudio
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

CHATML_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>"
CHATML_MID = "<|audio_end|>Transcribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"

TEST_CLEAN_ARROW = (
    "/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/cache/openslr___librispeech_asr/"
    "all/0.0.0/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/librispeech_asr-test.clean.arrow"
)

SR = 48000
HOP = 1920
MAX_NEW_TOKENS = 256


def load_testclean():
    table = ipc.open_stream(TEST_CLEAN_ARROW).read_all()
    ids = table.column("id").to_pylist()
    texts = table.column("text").to_pylist()
    audio = table.column("audio").to_pylist()
    rows = []
    for uid, text, a in zip(ids, texts, audio):
        rows.append({"id": uid, "text": text, "bytes": a["bytes"]})
    return rows


def preprocess_audio(wav_bytes: bytes) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # (S,)


def build_prompt_ids(tokenizer, audio_pad_id: int, t_audio: int):
    prefix = tokenizer.encode(CHATML_PREFIX, add_special_tokens=False)
    mid = tokenizer.encode(CHATML_MID, add_special_tokens=False)
    return prefix + [audio_pad_id] * t_audio + mid


def run_batch(model, tokenizer, cfg, batch):
    """batch: list of (id, text_ref, waveform_tensor_1d)"""
    audio_pad_id = cfg.audio_pad_token_id
    pad_id = cfg.pad_token_id
    eos_id = cfg.eos_token_id

    # Build prompts
    prompts = []
    t_audios = []
    for _, _, wav in batch:
        t_audio = wav.shape[-1] // HOP
        if t_audio == 0:
            t_audio = 1  # extremely short clips: pad one step
        t_audios.append(t_audio)
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio))

    # Left-pad prompts for generation
    max_len = max(len(p) for p in prompts)
    input_ids = []
    attn_mask = []
    for p in prompts:
        pad = max_len - len(p)
        input_ids.append([pad_id] * pad + p)
        attn_mask.append([0] * pad + [1] * len(p))
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=model.device)

    # Pack audio features: (N, 1, S_max)
    max_s = max(wav.shape[-1] for _, _, wav in batch)
    wav_tensors = []
    for _, _, wav in batch:
        pad = max_s - wav.shape[-1]
        wav_tensors.append(torch.nn.functional.pad(wav, (0, pad)))
    audio_features = torch.stack(wav_tensors, dim=0).unsqueeze(1).to(model.device, dtype=torch.bfloat16)
    audio_lengths = torch.tensor(t_audios, dtype=torch.long, device=model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            audio_features=audio_features,
            audio_lengths=audio_lengths,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )

    hyps = []
    for i in range(out.size(0)):
        gen = out[i, max_len:]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        hyps.append(txt)
    return hyps


def eval_checkpoint(ckpt_path: Path, rows, batch_size: int, max_samples: int, out_dir: Path,
                    normalizer: EnglishTextNormalizer):
    print(f"[eval] loading {ckpt_path}", flush=True)
    cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()

    # Patch the model's DynamicCache reference so the hybrid linear-attention path gets
    # the Qwen3Next cache (with has_previous_state / conv_states / recurrent_states).
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        if "modeling_qwen3_5AE" in mod_name and hasattr(mod, "DynamicCache"):
            mod.DynamicCache = Qwen3NextDynamicCache

    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    if max_samples is None or max_samples > len(rows):
        max_samples = len(rows)
    rows_sub = rows[:max_samples]

    # Sort by waveform length to stabilize batch shapes
    prepared = []
    for r in rows_sub:
        wav = preprocess_audio(r["bytes"])
        prepared.append({"id": r["id"], "text": r["text"], "wav": wav})
    prepared.sort(key=lambda x: x["wav"].shape[-1])

    hyps_by_id = {}
    t0 = time.time()
    with open(hyp_path, "w") as f:
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            batch = [(r["id"], r["text"], r["wav"]) for r in chunk]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Fall back to size 1
                hyps = []
                for one in batch:
                    hyps.extend(run_batch(model, tokenizer, cfg, [one]))
            for r, hyp in zip(chunk, hyps):
                rec = {"id": r["id"], "ref": r["text"], "hyp": hyp}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                hyps_by_id[r["id"]] = (r["text"], hyp)
            if (i // batch_size) % 20 == 0:
                done = i + len(chunk)
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(prepared) - done) / max(rate, 1e-6)
                print(f"[eval] {ckpt_path.name} {done}/{len(prepared)}  {rate:.2f} sps  eta {eta/60:.1f} min",
                      flush=True)

    # Compute WER
    refs_raw = [v[0] for v in hyps_by_id.values()]
    hyps_raw = [v[1] for v in hyps_by_id.values()]
    refs_norm = [normalizer(x).strip() for x in refs_raw]
    hyps_norm = [normalizer(x).strip() for x in hyps_raw]

    # Drop pairs where ref_norm is empty (jiwer crashes)
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r]
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]
    wer_norm = jiwer.wer(refs_f, hyps_f) if refs_f else float("nan")
    cer_norm = jiwer.cer(refs_f, hyps_f) if refs_f else float("nan")
    wer_raw = jiwer.wer(refs_raw, hyps_raw) if refs_raw else float("nan")

    summary = {
        "checkpoint": str(ckpt_path),
        "n_samples": len(rows_sub),
        "n_scored": len(pairs),
        "wer_normalized": wer_norm,
        "cer_normalized": cer_norm,
        "wer_raw": wer_raw,
        "elapsed_sec": time.time() - t0,
        "batch_size": batch_size,
        "normalizer": "whisper.EnglishTextNormalizer",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[eval] {ckpt_path.name} WER(norm)={wer_norm:.4f} CER(norm)={cer_norm:.4f} "
          f"WER(raw)={wer_raw:.4f} n={len(pairs)}", flush=True)

    # Release
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ckpts", type=str, default=None,
                   help="comma-separated checkpoint step numbers (e.g. 46000,45000). default = all")
    p.add_argument("--include-partial", action="store_true",
                   help="also evaluate checkpoints whose safetensors are still being written (mtime < 60s)")
    return p.parse_args()


def find_checkpoints(root: Path, steps_filter=None, include_partial=False):
    ckpts = []
    for p in sorted(root.iterdir()):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if steps_filter and step not in steps_filter:
            continue
        st = p / "model.safetensors.index.json"
        if not st.exists():
            # Might be monolithic safetensors (rare here) — fall back
            st = p / "model.safetensors"
        if not st.exists():
            print(f"[eval] skip {p.name} (no safetensors yet)", flush=True)
            continue
        mtime = st.stat().st_mtime
        if not include_partial and (time.time() - mtime) < 60:
            print(f"[eval] skip {p.name} (save in progress)", flush=True)
            continue
        ckpts.append((step, p))
    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]


def main():
    args = parse_args()
    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    steps_filter = None
    if args.ckpts:
        steps_filter = {int(s) for s in args.ckpts.split(",")}

    ckpts = find_checkpoints(ckpt_root, steps_filter, include_partial=args.include_partial)
    if not ckpts:
        print("[eval] no checkpoints found", flush=True)
        sys.exit(1)

    print(f"[eval] {len(ckpts)} checkpoints, loading test-clean", flush=True)
    rows = load_testclean()
    print(f"[eval] {len(rows)} test-clean rows", flush=True)

    normalizer = EnglishTextNormalizer({})

    all_summaries = []
    for p in ckpts:
        ckpt_out = out_root / p.name
        summary_path = ckpt_out / "summary.json"
        if summary_path.exists():
            print(f"[eval] {p.name} already done, skipping (delete {summary_path} to re-run)", flush=True)
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue
        try:
            summary = eval_checkpoint(p, rows, args.batch_size, args.max_samples, ckpt_out, normalizer)
            all_summaries.append(summary)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[eval] ERROR on {p.name}: {e}", flush=True)

    # Global summary
    global_summary = out_root / "summary_all.json"
    with open(global_summary, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote {global_summary}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'checkpoint':<40s}  {'WER(norm)':>10s}  {'CER(norm)':>10s}  {'WER(raw)':>10s}  {'n':>5s}")
    for s in all_summaries:
        name = Path(s["checkpoint"]).name
        print(f"{name:<40s}  {s['wer_normalized']:>10.4f}  {s['cer_normalized']:>10.4f}  "
              f"{s['wer_raw']:>10.4f}  {s['n_scored']:>5d}")


if __name__ == "__main__":
    main()
