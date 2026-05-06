"""Teacher-forced WER evaluation.

Instead of autoregressive generate(), this script runs a single forward pass
with the ground-truth reference tokens as input (teacher-forcing), collects
per-position argmax predictions for the generation span, and computes WER.

If teacher-forced WER ≈ 0% but autoregressive WER is high → exposure bias /
the model has learned the mapping but cannot generate autoregressively.
If teacher-forced WER is also high → the model itself is broken (audio encoder
not producing useful features).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n audiollm python evaluation/eval_teacher_forced_wer.py \
        --ckpt-root .../checkpoint-7000 \
        --max-samples 200 \
        --out-root /mnt/tmp/eval_tf_ckpt7000
"""

import argparse
import ctypes
import io
import json
import os
import time
from pathlib import Path

_stub = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib", "glibc_stub.so")
if os.path.exists(_stub):
    ctypes.CDLL(_stub, mode=ctypes.RTLD_GLOBAL)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import jiwer
import pyarrow.ipc as ipc
import torch
import torchaudio
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

CHATML_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>"
CHATML_MID = "<|audio_end|>Transcribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"
CHATML_EOS = "<|im_end|>"

TEST_CLEAN_ARROW = (
    "/mnt/tmp/cache/openslr___librispeech_asr/all/0.0.0/"
    "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/librispeech_asr-test.clean.arrow"
)

SR = 24000
HOP = 1920


def load_arrow(path):
    table = ipc.open_stream(path).read_all()
    ids = table.column("id").to_pylist()
    texts = table.column("text").to_pylist()
    audio = table.column("audio").to_pylist()
    return [{"id": uid, "text": t, "bytes": a["bytes"]} for uid, t, a in zip(ids, texts, audio)]


def preprocess_audio(wav_bytes):
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)


def teacher_forced_batch(model, tokenizer, cfg, batch):
    """
    batch: list of (id, ref_text, waveform_1d)
    Returns list of (hyp_text, ref_text) decoded from teacher-forced predictions.
    """
    audio_pad_id = cfg.audio_pad_token_id
    pad_id = cfg.pad_token_id
    eos_id = cfg.eos_token_id

    prefix_ids = tokenizer.encode(CHATML_PREFIX, add_special_tokens=False)
    mid_ids = tokenizer.encode(CHATML_MID, add_special_tokens=False)
    eos_ids = tokenizer.encode(CHATML_EOS, add_special_tokens=False)

    t_audios = []
    ref_token_lists = []
    full_seqs = []

    for _, ref_text, wav in batch:
        t_audio = max(1, wav.shape[-1] // HOP)
        t_audios.append(t_audio)

        ref_ids = tokenizer.encode(ref_text, add_special_tokens=False) + eos_ids
        ref_token_lists.append(ref_ids)

        seq = prefix_ids + [audio_pad_id] * t_audio + mid_ids + ref_ids
        full_seqs.append(seq)

    # Right-pad to max length
    max_len = max(len(s) for s in full_seqs)
    input_ids_list, attn_mask_list, gen_start_list = [], [], []
    for seq, ref_ids, t_audio in zip(full_seqs, ref_token_lists, t_audios):
        gen_start = len(prefix_ids) + t_audio + len(mid_ids)
        gen_start_list.append(gen_start)
        pad_len = max_len - len(seq)
        input_ids_list.append(seq + [pad_id] * pad_len)
        attn_mask_list.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=model.device)
    attn_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=model.device)

    # Pack audio features
    max_s = max(wav.shape[-1] for _, _, wav in batch)
    wav_tensors = []
    for _, _, wav in batch:
        pad = max_s - wav.shape[-1]
        wav_tensors.append(torch.nn.functional.pad(wav, (0, pad)))
    audio_features = torch.stack(wav_tensors).unsqueeze(1).to(model.device, dtype=torch.bfloat16)
    audio_lengths = torch.tensor(t_audios, dtype=torch.long, device=model.device)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            audio_features=audio_features,
            audio_lengths=audio_lengths,
        )

    logits = out.logits  # (B, T, V)

    results = []
    for i, (ref_ids, gen_start) in enumerate(zip(ref_token_lists, gen_start_list)):
        # predictions at positions gen_start-1 .. gen_start+len(ref_ids)-2
        # (model predicts token t+1 from position t)
        pred_slice = logits[i, gen_start - 1 : gen_start + len(ref_ids) - 1]  # (L, V)
        pred_ids = pred_slice.argmax(dim=-1).tolist()
        # strip from first predicted EOS
        hyp_ids = []
        for tid in pred_ids:
            if tid == cfg.eos_token_id:
                break
            hyp_ids.append(tid)
        hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True).strip()
        ref_text_decoded = tokenizer.decode(
            [t for t in ref_ids if t not in tokenizer.all_special_ids], skip_special_tokens=True
        ).strip()
        results.append((hyp_text, ref_text_decoded))
    return results


def run_eval(ckpt_path: Path, rows, batch_size, max_samples, out_dir,
             normalizer: EnglishTextNormalizer):
    print(f"[tf-eval] loading {ckpt_path}", flush=True)

    adapter_cfg_path = ckpt_path / "adapter_config.json"
    if adapter_cfg_path.exists():
        base_path = json.loads(adapter_cfg_path.read_text())["base_model_name_or_path"]
        cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        base = AutoModel.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(ckpt_path)).cuda().eval()
    else:
        cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True,
                                          attn_implementation="sdpa").cuda().eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    if max_samples:
        rows = rows[:max_samples]

    from tqdm import tqdm
    prepared = []
    for r in tqdm(rows, desc="preprocess", unit="sample", dynamic_ncols=True):
        wav = preprocess_audio(r["bytes"])
        prepared.append({"id": r["id"], "text": r["text"], "wav": wav})
    prepared.sort(key=lambda x: x["wav"].shape[-1])

    hyps_all, refs_all = [], []
    t0 = time.time()
    pbar = tqdm(total=len(prepared), desc="teacher-forcing", unit="sample", dynamic_ncols=True)
    pred_path = out_dir / "predictions_tf.jsonl"
    with open(pred_path, "w") as f:
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            batch = [(r["id"], r["text"], r["wav"]) for r in chunk]
            try:
                results = teacher_forced_batch(model, tokenizer, cfg, batch)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                results = []
                for one in batch:
                    results.extend(teacher_forced_batch(model, tokenizer, cfg, [one]))
            for r, (hyp, ref) in zip(chunk, results):
                f.write(json.dumps({"id": r["id"], "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
                hyps_all.append(hyp)
                refs_all.append(ref)
            pbar.update(len(chunk))
    pbar.close()

    refs_norm = [normalizer(x).strip() for x in refs_all]
    hyps_norm = [normalizer(x).strip() for x in hyps_all]
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r]
    wer_norm = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
    cer_norm = jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])
    wer_raw  = jiwer.wer(refs_all, hyps_all) if refs_all else float("nan")

    summary = {
        "checkpoint": str(ckpt_path),
        "mode": "teacher_forced",
        "n_samples": len(prepared),
        "n_scored": len(pairs),
        "wer_normalized": wer_norm,
        "cer_normalized": cer_norm,
        "wer_raw": wer_raw,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "summary_tf.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[tf-eval] WER(norm)={wer_norm:.4f}  CER(norm)={cer_norm:.4f}  "
          f"WER(raw)={wer_raw:.4f}  n={len(pairs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-clean-arrow", default=TEST_CLEAN_ARROW)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    normalizer = EnglishTextNormalizer({})

    print(f"[tf-eval] loading data from {args.test_clean_arrow}", flush=True)
    rows = load_arrow(args.test_clean_arrow)
    print(f"[tf-eval] {len(rows)} rows loaded", flush=True)

    run_eval(ckpt_path, rows, args.batch_size, args.max_samples, out_root, normalizer)


if __name__ == "__main__":
    main()
