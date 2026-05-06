"""LibriSpeech test-clean WER sweep for Whisper-small Stage1 checkpoints.

Differences from eval_testclean_wer.py (DAC/48k version):
  * Audio resampled to 16 kHz (Whisper SR)
  * audio_features = [N, 80, 3000] log-mel via WhisperFeatureExtractor  (NOT raw waveform)
  * audio_lengths  = ceil(num_samples / 320), capped at 1500            (Whisper 50 fps × 30 s)

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_testclean_wer_whisper.py \
        --ckpt-root /mnt/.../Qwen3.5_whisper_small_Stage1/Qwen3.5_whisper_small_Stage1 \
        --out-root  /mnt/.../eval_testclean_whisper \
        [--max-samples 2620] [--batch-size 4] [--ckpts 11000]

    # Single checkpoint (no sweep):
    CUDA_VISIBLE_DEVICES=0 python eval_testclean_wer_whisper.py \
        --ckpt-root /path/to/parent_of_checkpoint_dirs \
        --out-root  ./eval_out \
        --ckpts 11000
"""

import argparse
import ctypes
import io
import json
import math
import os
import re
import sys

# flash_attn requires GLIBC_2.32 (__libc_single_threaded).  On EL7 systems the
# symbol is provided by glibc_stub.so (built by install_env.sh).  Load it via
# ctypes before any flash_attn import so the dynamic linker can resolve it.
_stub = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib", "glibc_stub.so")
if os.path.exists(_stub):
    ctypes.CDLL(_stub, mode=ctypes.RTLD_GLOBAL)
import time
from pathlib import Path

import jiwer
import pyarrow.ipc as ipc
import torch
import torchaudio
from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperFeatureExtractor
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# ── Compatibility patch ────────────────────────────────────────────────────────
# modeling_qwen3_5AE was written against an older Qwen3NextDynamicCache API.
# Bridge the gap with a subclass that re-exposes the old interface.

class _LayerState:
    """Per-layer state proxy so cache_params.layers[i].conv_states works."""
    def __init__(self, cache, idx):
        self._c, self._i = cache, idx

    @property
    def conv_states(self):
        return self._c.conv_states[self._i]

    @property
    def recurrent_states(self):
        return self._c.recurrent_states[self._i]


class _PatchedCache(Qwen3NextDynamicCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = [_LayerState(self, i) for i in range(len(self.conv_states))]

    # Old API: called as a method with optional layer_idx arg.
    def has_previous_state(self, _layer_idx=None):
        return self.conv_states[self.last_linear_layer] is not None

    def update_conv_state(self, conv_state, layer_idx):
        self.conv_states[layer_idx] = conv_state
        return conv_state

    def update_recurrent_state(self, recurrent_state, layer_idx):
        self.recurrent_states[layer_idx] = recurrent_state

# ── Whisper constants ──────────────────────────────────────────────────────────
WHISPER_SR = 16_000
WHISPER_HOP = 320        # audio samples per encoder output frame
WHISPER_MAX_FRAMES = 1500
WHISPER_MEL_BINS = 80
WHISPER_MEL_FRAMES = 3000
WHISPER_MAX_SAMPLES = 30 * WHISPER_SR  # 480 000
WHISPER_MODEL_ID = "openai/whisper-small.en"

MAX_NEW_TOKENS = 256

CHATML_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>"
CHATML_MID = "<|audio_end|>Transcribe the audio to text.<|im_end|>\n<|im_start|>assistant\n"

TEST_CLEAN_ARROW = (
    # "/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/cache/openslr___librispeech_asr/"
    # "all/0.0.0/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/librispeech_asr-test.clean.arrow"
    "/mnt/tmp/cache/openslr___librispeech_asr/all/0.0.0/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/librispeech_asr-test.clean.arrow"
)

# Lazily initialised once per process.
_FE: WhisperFeatureExtractor | None = None


def get_fe() -> WhisperFeatureExtractor:
    global _FE
    if _FE is None:
        _FE = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_ID)
    return _FE


# ── Data loading ───────────────────────────────────────────────────────────────

def load_testclean():
    table = ipc.open_stream(TEST_CLEAN_ARROW).read_all()
    ids = table.column("id").to_pylist()
    texts = table.column("text").to_pylist()
    audio = table.column("audio").to_pylist()
    return [{"id": uid, "text": text, "bytes": a["bytes"]} for uid, text, a in zip(ids, texts, audio)]


DAC_SR = 48_000
DAC_HOP = 1_920  # samples per encoder output frame at 48 kHz


def _is_whisper_config(cfg) -> bool:
    return hasattr(cfg, "audio_config") and hasattr(cfg.audio_config, "whisper_model_id")


def _is_wavtok_config(cfg) -> bool:
    return hasattr(cfg, "audio_config") and hasattr(cfg.audio_config, "wavtok_sample_rate")


def _raw_waveform_sr_hop(cfg) -> tuple[int, int]:
    """Return (sample_rate, hop_length) for raw-waveform encoders (DAC / WavTok)."""
    if _is_wavtok_config(cfg):
        return cfg.audio_config.wavtok_sample_rate, cfg.audio_config.wavtok_hop_length
    return DAC_SR, DAC_HOP


def preprocess_audio(wav_bytes: bytes, target_sr: int) -> torch.Tensor:
    """Load bytes → mono float32 waveform at target_sr, shape (S,)."""
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # (S,)


def extract_mel(wav: torch.Tensor) -> torch.Tensor:
    """1-D float32 waveform (16 kHz, ≤30 s) → [80, 3000] log-mel (float32)."""
    if wav.numel() > WHISPER_MAX_SAMPLES:
        wav = wav[:WHISPER_MAX_SAMPLES]
    fe = get_fe()
    mel = fe(wav.numpy(), sampling_rate=WHISPER_SR, return_tensors="pt").input_features
    return mel.squeeze(0)  # [80, 3000]


def audio_pad_token_count(num_samples: int) -> int:
    if num_samples <= 0:
        return 0
    return min(WHISPER_MAX_FRAMES, math.ceil(num_samples / WHISPER_HOP))


# ── Prompt construction ────────────────────────────────────────────────────────

def build_prompt_ids(tokenizer, audio_pad_id: int, t_audio: int) -> list[int]:
    prefix = tokenizer.encode(CHATML_PREFIX, add_special_tokens=False)
    mid = tokenizer.encode(CHATML_MID, add_special_tokens=False)
    return prefix + [audio_pad_id] * t_audio + mid


# ── Batch inference ────────────────────────────────────────────────────────────

def run_batch(model, tokenizer, cfg, batch):
    """
    batch: list of (id, text_ref, wav_1d_tensor)

    Whisper encoder: audio_features [N, 80, 3000] float32, audio_lengths ceil(S/320) ≤ 1500
    DAC encoder:     audio_features [N, 1, S_max]  bfloat16, audio_lengths S // 1920
    """
    use_whisper = _is_whisper_config(cfg)
    raw_sr, raw_hop = _raw_waveform_sr_hop(cfg)
    audio_pad_id = cfg.audio_pad_token_id
    pad_id = cfg.pad_token_id
    eos_id = cfg.eos_token_id

    prompts, audio_tensors, t_audios = [], [], []
    for _, _, wav in batch:
        n_samples = wav.shape[-1]
        if use_whisper:
            t_audio = audio_pad_token_count(n_samples)
            audio_tensors.append(extract_mel(wav))             # [80, 3000]
        else:
            t_audio = n_samples // raw_hop
            if t_audio == 0:
                t_audio = 1
            audio_tensors.append(wav)                          # (S,)
        t_audios.append(t_audio)
        prompts.append(build_prompt_ids(tokenizer, audio_pad_id, t_audio))

    # Left-pad token sequences for batch generation.
    max_len = max(len(p) for p in prompts)
    input_ids, attn_mask = [], []
    for p in prompts:
        pad = max_len - len(p)
        input_ids.append([pad_id] * pad + p)
        attn_mask.append([0] * pad + [1] * len(p))
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=model.device)

    if use_whisper:
        # [N, 80, 3000] float32 — cast to bf16 inside AudioEncoder
        audio_features = torch.stack(audio_tensors, dim=0).to(model.device)
    else:
        # [N, 1, S_max] bfloat16 raw waveform (DAC 48kHz or WavTok 24kHz)
        max_s = max(w.shape[-1] for w in audio_tensors)
        audio_features = torch.stack(
            [torch.nn.functional.pad(w, (0, max_s - w.shape[-1])) for w in audio_tensors]
        ).unsqueeze(1).to(model.device, dtype=torch.bfloat16)

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
        hyps.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return hyps


# ── Per-checkpoint evaluation ──────────────────────────────────────────────────

def eval_checkpoint(ckpt_path: Path, rows, batch_size: int, max_samples: int,
                    out_dir: Path, normalizer: EnglishTextNormalizer):
    print(f"[eval] loading {ckpt_path}", flush=True)
    cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()

    # Patch DynamicCache so the hybrid linear-attention path uses _PatchedCache,
    # which makes has_previous_state() callable (it's a @property in newer transformers).
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        if "modeling_qwen3_5AE" in mod_name and hasattr(mod, "DynamicCache"):
            mod.DynamicCache = _PatchedCache

    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    if max_samples is None or max_samples > len(rows):
        max_samples = len(rows)
    rows_sub = rows[:max_samples]

    target_sr = WHISPER_SR if _is_whisper_config(cfg) else _raw_waveform_sr_hop(cfg)[0]

    # Pre-load and sort by waveform length to stabilise batch padding.
    prepared = []
    for r in rows_sub:
        wav = preprocess_audio(r["bytes"], target_sr)
        prepared.append({"id": r["id"], "text": r["text"], "wav": wav})
    prepared.sort(key=lambda x: x["wav"].shape[-1])

    hyps_by_id = {}
    t0 = time.time()
    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            batch_in = [(r["id"], r["text"], r["wav"]) for r in chunk]
            try:
                hyps = run_batch(model, tokenizer, cfg, batch_in)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyps = []
                for one in batch_in:
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
                print(f"[eval] {ckpt_path.name}  {done}/{len(prepared)}  "
                      f"{rate:.2f} sps  eta {eta/60:.1f} min", flush=True)

    refs_raw = [v[0] for v in hyps_by_id.values()]
    hyps_raw = [v[1] for v in hyps_by_id.values()]
    refs_norm = [normalizer(x).strip() for x in refs_raw]
    hyps_norm = [normalizer(x).strip() for x in hyps_raw]

    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r]
    refs_f, hyps_f = [r for r, _ in pairs], [h for _, h in pairs]
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
        "audio_sr": WHISPER_SR if _is_whisper_config(cfg) else _raw_waveform_sr_hop(cfg)[0],
        "audio_format": "log-mel [80, 3000]" if _is_whisper_config(cfg) else "raw [N, 1, S] bf16",
        "normalizer": "whisper.EnglishTextNormalizer",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[eval] {ckpt_path.name}  WER(norm)={wer_norm:.4f}  CER(norm)={cer_norm:.4f}  "
          f"WER(raw)={wer_raw:.4f}  n={len(pairs)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def _has_safetensors(p: Path) -> bool:
    return (p / "model.safetensors.index.json").exists() or (p / "model.safetensors").exists()


def find_checkpoints(root: Path, steps_filter=None, include_partial=False):
    # Case 1: root itself is a checkpoint (safetensors live directly in root).
    if _has_safetensors(root):
        return [root]

    ckpts = []
    for p in sorted(root.iterdir()):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if steps_filter and step not in steps_filter:
            continue
        if not _has_safetensors(p):
            print(f"[eval] skip {p.name} (no safetensors yet)", flush=True)
            continue
        st = p / "model.safetensors.index.json"
        if not st.exists():
            st = p / "model.safetensors"
        mtime = st.stat().st_mtime
        if not include_partial and (time.time() - mtime) < 60:
            print(f"[eval] skip {p.name} (save in progress)", flush=True)
            continue
        ckpts.append((step, p))
    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True,
                   help="directory containing checkpoint-N subdirs (or a single checkpoint dir)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ckpts", type=str, default=None,
                   help="comma-separated step numbers, e.g. 11000,10000")
    p.add_argument("--include-partial", action="store_true")
    p.add_argument("--test-clean-arrow", type=str, default=TEST_CLEAN_ARROW,
                   help="path to librispeech_asr-test.clean.arrow")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow overriding the arrow path via CLI.
    global TEST_CLEAN_ARROW
    TEST_CLEAN_ARROW = args.test_clean_arrow

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

    print(f"[eval] {len(ckpts)} checkpoint(s) to evaluate, loading test-clean …", flush=True)
    rows = load_testclean()
    print(f"[eval] {len(rows)} test-clean rows loaded", flush=True)

    normalizer = EnglishTextNormalizer({})
    all_summaries = []

    for p in ckpts:
        ckpt_out = out_root / p.name
        summary_path = ckpt_out / "summary.json"
        if summary_path.exists():
            print(f"[eval] {p.name} already done — delete {summary_path} to re-run", flush=True)
            with open(summary_path) as f:
                all_summaries.append(json.load(f))
            continue
        try:
            summary = eval_checkpoint(p, rows, args.batch_size, args.max_samples, ckpt_out, normalizer)
            all_summaries.append(summary)
        except Exception:
            import traceback
            traceback.print_exc()

    global_summary = out_root / "summary_all.json"
    with open(global_summary, "w", encoding="utf-8") as f:
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
