"""
Layer-wise embedding extraction for Representation Richness analysis.

Extracts 37 layers per ALM family:
  encoder_out (1) + projector_L1..L4 (4) + llm_L01..L32 (32)

Per layer, per utterance → {utt_id}.npz with keys:
  mean   (d,) float32
  last   (d,) float32  (projector + LLM only)
  frames (T_ds, d) float16
  n_valid int

Batched + DataLoader-prefetched for throughput. Saves npz via background thread.

Usage:
  python extract.py --family whisper_tiny --manifest /mnt/tmp/cache/cmu_arctic_7/manifest.csv \
      --out_dir experiments/representation_richness/cmu_arctic_7 \
      [--batch_size 8] [--num_workers 4]
      [--speakers bdl,slt --max_utt_per_speaker 2]
"""

import argparse
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from experiments.representation_richness.load_alm import load_alm, t_audio_for_family

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_FRAMES = 256  # max frame-level frames stored per utterance per layer

# ── Whisper feature extractor cache ──────────────────────────────────────────

_whisper_fe: dict = {}

def _get_whisper_fe(model_id: str):
    if model_id not in _whisper_fe:
        from transformers import WhisperFeatureExtractor
        _whisper_fe[model_id] = WhisperFeatureExtractor.from_pretrained(model_id)
    return _whisper_fe[model_id]


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    Loads and preprocesses audio for one ALM family.
    Returns one item per utterance: (utt_id, audio_tensor, n_samples, t_audio).
    Audio tensor is:
      Whisper → (80, 3000) log-mel float32
      WavTok/DACVAE → (1, S) raw waveform float32 at family sample_rate
    """

    def __init__(self, rows: list[dict], cfg: dict):
        self.rows = rows
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        utt_id = str(row["utt_id"])
        path = str(row["audio_path"])

        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            return {"utt_id": utt_id, "error": str(e)}

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_sr = self.cfg["sample_rate"]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        n_samples = wav.shape[-1]
        t_audio = t_audio_for_family(self.cfg, n_samples)
        if t_audio == 0:
            return {"utt_id": utt_id, "error": "t_audio=0"}

        if self.cfg["encoder_type"] == "whisper":
            fe = _get_whisper_fe(self.cfg["whisper_model_id"])
            mel = fe(wav.squeeze(0).numpy(), sampling_rate=16000,
                     return_tensors="pt").input_features  # (1, 80, 3000)
            audio_tensor = mel.squeeze(0)  # (80, 3000)
        else:
            audio_tensor = wav  # (1, S)

        return {
            "utt_id": utt_id,
            "audio_tensor": audio_tensor,
            "n_samples": n_samples,
            "t_audio": t_audio,
        }


def _collate(batch: list[dict]) -> dict:
    """Collate a batch of AudioDataset items, skipping errors."""
    valid = [b for b in batch if "error" not in b]
    if not valid:
        return {"valid": [], "errors": [b["utt_id"] for b in batch if "error" in b]}

    utt_ids = [b["utt_id"] for b in valid]
    t_audios = [b["t_audio"] for b in valid]
    n_samples = [b["n_samples"] for b in valid]
    tensors = [b["audio_tensor"] for b in valid]

    # Stack audio tensors; wavtok/dacvae need zero-padding to same length
    if tensors[0].ndim == 2 and tensors[0].shape[0] in (80,):
        # Whisper mel: all (80, 3000) → easy stack
        audio_batch = torch.stack(tensors, dim=0)  # (B, 80, 3000)
    else:
        # Raw waveform (1, S): pad to max S
        max_s = max(t.shape[-1] for t in tensors)
        audio_batch = torch.stack(
            [F.pad(t, (0, max_s - t.shape[-1])) for t in tensors], dim=0
        )  # (B, 1, max_s)

    return {
        "valid": valid,
        "utt_ids": utt_ids,
        "t_audios": t_audios,
        "n_samples": n_samples,
        "audio_batch": audio_batch,
        "errors": [b["utt_id"] for b in batch if "error" in b],
    }


# ── Hook capture ──────────────────────────────────────────────────────────────

class HookCapture:
    def __init__(self):
        self.captured: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, module: torch.nn.Module, name: str):
        def hook(_, _input, output):
            if isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, "last_hidden_state"):
                h = output.last_hidden_state
            else:
                h = output
            # Keep on GPU as float16 — move to CPU after batch completes
            self.captured[name] = h.detach().half()
        self._handles.append(module.register_forward_hook(hook))

    def to_cpu_float32(self):
        """Move all captured tensors to CPU float32 for processing."""
        return {k: v.float().cpu() for k, v in self.captured.items()}

    def clear(self):
        self.captured.clear()

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


LLM_INTERVAL = 3  # sample every Nth LLM layer: L01, L04, L07, ..., L31


def _register_hooks(model, cfg: dict, capture: HookCapture) -> list[str]:
    """
    Register hooks on 16 layers (interval-3 sampling):
      encoder_out (1) + projector_L1..L4 (4) + llm L01,L04,...,L31 (11)
    """
    inner = model.model
    layer_names = []

    if cfg["encoder_type"] in ("whisper", "wavtok"):
        capture.register(inner.audio_encoder.encoder, "encoder_out")
        layer_names.append("encoder_out")

    # All projector layers (only 4)
    for i, layer in enumerate(inner.audio_encoder.projector.layers):
        name = f"projector_L{i+1}"
        capture.register(layer, name)
        layer_names.append(name)

    # LLM: every LLM_INTERVAL layers (1-indexed: L01, L04, L07, ..., L31)
    selected_llm = list(range(1, len(inner.language_model.layers) + 1, LLM_INTERVAL))
    for i, layer in enumerate(inner.language_model.layers):
        layer_num = i + 1
        if layer_num in selected_llm:
            name = f"llm_L{layer_num:02d}"
            capture.register(layer, name)
            layer_names.append(name)

    logger.info(f"Hooks: {len(layer_names)} total — {layer_names}")
    return layer_names


# ── Input construction ────────────────────────────────────────────────────────

_enc_cache: dict = {}  # tokenizer id → enc result cache for fixed strings

def _build_batch_inputs(tokenizer, cfg: dict, audio_batch: torch.Tensor,
                        t_audios: list[int], device: str):
    """
    Build padded input_ids for a batch with different t_audio per sample.

    Returns:
      input_ids  (B, max_len)
      attn_mask  (B, max_len)
      audio_feats  (B, ...) on device
      audio_lengths (B,)
      per_sample_audio_masks  list[B] of (T_len,) bool tensors (CPU)
    """
    tok_id = id(tokenizer)
    if tok_id not in _enc_cache:
        def enc(text):
            return tokenizer.encode(text, add_special_tokens=False)
        sys_user = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        end_user = "<|im_end|>\n<|im_start|>assistant\n"
        _enc_cache[tok_id] = {
            "sys_user": enc(sys_user),
            "audio_start": enc("<|audio_start|>"),
            "audio_end": enc("<|audio_end|>"),
            "end_user": enc(end_user),
            "audio_pad_id": tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
        }
    c = _enc_cache[tok_id]
    prefix = c["sys_user"] + c["audio_start"]
    suffix = c["audio_end"] + c["end_user"]
    apid = c["audio_pad_id"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    seqs = [prefix + [apid] * t + suffix for t in t_audios]
    max_len = max(len(s) for s in seqs)

    input_ids_list = []
    attn_masks = []
    per_sample_audio_masks = []

    for seq in seqs:
        pad_len = max_len - len(seq)
        padded = seq + [pad_id] * pad_len
        attn = [1] * len(seq) + [0] * pad_len
        input_ids_list.append(padded)
        attn_masks.append(attn)
        audio_mask = torch.tensor([x == apid for x in padded], dtype=torch.bool)
        per_sample_audio_masks.append(audio_mask)

    input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attn_mask_t = torch.tensor(attn_masks, dtype=torch.long, device=device)
    audio_lengths_t = torch.tensor(t_audios, dtype=torch.long, device=device)

    # Audio features: move to device; for whisper (B, 80, 3000) else (B, 1, S)
    audio_feats_t = audio_batch.to(device)
    if audio_feats_t.dim() == 3 and audio_feats_t.shape[1] not in (1,):
        # (B, 80, 3000) mel — keep as-is
        pass
    else:
        # (B, 1, S) waveform — keep as-is
        pass

    return input_ids_t, attn_mask_t, audio_feats_t, audio_lengths_t, per_sample_audio_masks


# ── Per-sample layer processing ───────────────────────────────────────────────

def _downsample(h: np.ndarray) -> np.ndarray:
    T = h.shape[0]
    if T <= MAX_FRAMES:
        return h.astype(np.float16)
    idx = np.round(np.linspace(0, T - 1, MAX_FRAMES)).astype(int)
    return h[idx].astype(np.float16)


def _process_sample(
    name: str,
    h_b: torch.Tensor,      # (T, D) float32 CPU — already sliced to sample b
    t_audio: int,
    audio_mask_b: torch.Tensor,  # (T_total,) bool CPU
    is_encoder_out: bool,
    encoder_type: str,
) -> dict | None:
    """Extract mean/last/frames for one sample, one layer."""
    T, D = h_b.shape

    # WavTok encoder output is channels-first [D, T] — transpose
    if is_encoder_out and encoder_type == "wavtok":
        h_b = h_b.T  # (T, D)
        T, D = h_b.shape

    # Select valid frames
    if is_encoder_out or "projector" in name:
        # encoder and projector: first t_audio frames are valid
        valid = h_b[:t_audio]
    else:
        # LLM: audio positions identified by audio_mask in padded sequence
        if h_b.shape[0] != audio_mask_b.shape[0]:
            valid = h_b[:t_audio]
        else:
            valid = h_b[audio_mask_b]

    if valid.shape[0] == 0:
        return None

    mean_v = valid.mean(dim=0).numpy().astype(np.float32)
    frames_v = _downsample(valid.numpy())

    result = {
        "mean": mean_v,
        "frames": frames_v,
        "n_valid": np.array(valid.shape[0], dtype=np.int32),
    }
    if not is_encoder_out:
        result["last"] = valid[-1].numpy().astype(np.float32)

    return result


# ── DACVAE encoder_out ────────────────────────────────────────────────────────

@torch.no_grad()
def _dacvae_encoder_out_batch(model, audio_feats: torch.Tensor, t_audios: list[int]) -> list[dict | None]:
    """Compute DACVAE encoder_out for a batch."""
    dac = model.model.audio_encoder.encoder
    try:
        result = dac.encode(audio_feats)
        z = result[0] if isinstance(result, tuple) else result
        # z: (B, D, T) → transpose per sample
    except Exception as e:
        logger.warning(f"DACVAE batch encode failed: {e}")
        return [None] * len(t_audios)

    z_cpu = z.float().cpu()  # (B, D, T)
    out = []
    for b, t in enumerate(t_audios):
        zb = z_cpu[b].T  # (T, D)
        valid = zb[:t]
        if valid.shape[0] == 0:
            out.append(None)
            continue
        out.append({
            "mean": valid.mean(dim=0).numpy().astype(np.float32),
            "frames": _downsample(valid.numpy()),
            "n_valid": np.array(valid.shape[0], dtype=np.int32),
        })
    return out


# ── File I/O ──────────────────────────────────────────────────────────────────

def _save_npz(out_dir: Path, utt_id: str, layer_name: str, data: dict):
    layer_dir = out_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(str(layer_dir / f"{utt_id}.npz"), **save_dict)


def _already_done(out_dir: Path, utt_id: str) -> bool:
    return (out_dir / "encoder_out" / f"{utt_id}.npz").exists()


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_family(
    family: str,
    manifest_path: str,
    out_dir: str,
    speakers: set | None = None,
    max_utt_per_speaker: int | None = None,
    device: str = "cuda:0",
    batch_size: int = 8,
    num_workers: int = 4,
):
    out_dir = Path(out_dir) / family
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, cfg = load_alm(family, device=device, dtype=torch.bfloat16)

    capture = HookCapture()
    layer_names = _register_hooks(model, cfg, capture)
    if "encoder_out" not in layer_names:
        layer_names = ["encoder_out"] + layer_names

    # Load & filter manifest
    df = pd.read_csv(manifest_path)
    if speakers:
        df = df[df["speaker_id"].isin(speakers)]
    if max_utt_per_speaker:
        df = df.groupby("speaker_id", group_keys=False).head(max_utt_per_speaker)

    # Filter already-done utterances
    rows = [r for r in df.to_dict("records") if not _already_done(out_dir, str(r["utt_id"]))]
    logger.info(f"Utterances to process: {len(rows)} (skipping already-done)")

    dataset = AudioDataset(rows, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )

    processed = skipped = failed = 0
    start = time.time()
    total = len(rows)

    executor = ThreadPoolExecutor(max_workers=4)

    def _save_results(utt_id, sample_results):
        for lname, data in sample_results.items():
            if data is not None:
                _save_npz(out_dir, utt_id, lname, data)

    for batch_idx, batch in enumerate(loader):
        errors = batch.get("errors", [])
        failed += len(errors)

        valid_items = batch.get("valid", [])
        if not valid_items:
            continue

        utt_ids = batch["utt_ids"]
        t_audios = batch["t_audios"]
        audio_batch = batch["audio_batch"]

        # Build batch model inputs
        input_ids, attn_mask, audio_feats, audio_lengths, audio_masks = _build_batch_inputs(
            tokenizer, cfg, audio_batch, t_audios, device
        )

        # Forward pass
        capture.clear()
        try:
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    audio_features=audio_feats,
                    audio_lengths=audio_lengths,
                )
        except Exception as e:
            logger.warning(f"Batch forward failed: {e}")
            failed += len(utt_ids)
            continue

        # Move captured tensors to CPU float32 once per batch
        captured_cpu = capture.to_cpu_float32()

        # DACVAE encoder_out: separate call
        dacvae_enc_outs = None
        if cfg["encoder_type"] == "dacvae":
            dacvae_enc_outs = _dacvae_encoder_out_batch(model, audio_feats, t_audios)

        # Process per-sample
        B = len(utt_ids)
        for b in range(B):
            sample_results = {}

            # encoder_out
            if cfg["encoder_type"] == "dacvae":
                sample_results["encoder_out"] = dacvae_enc_outs[b] if dacvae_enc_outs else None
            elif "encoder_out" in captured_cpu:
                h = captured_cpu["encoder_out"]  # (B, T, D) or (B, D, T) for wavtok
                hb = h[b]
                sample_results["encoder_out"] = _process_sample(
                    "encoder_out", hb, t_audios[b], audio_masks[b],
                    is_encoder_out=True, encoder_type=cfg["encoder_type"]
                )

            # Projector + LLM
            for name in layer_names:
                if name == "encoder_out":
                    continue
                if name not in captured_cpu:
                    continue
                h = captured_cpu[name]  # (B, T, D)
                hb = h[b]
                sample_results[name] = _process_sample(
                    name, hb, t_audios[b], audio_masks[b],
                    is_encoder_out=False, encoder_type=cfg["encoder_type"]
                )

            # Save via thread pool
            executor.submit(_save_results, utt_ids[b], sample_results)
            processed += 1

        # Progress every 10 batches
        done_so_far = processed + skipped + failed
        if (batch_idx + 1) % 10 == 0 or done_so_far >= total:
            elapsed = time.time() - start
            rate = processed / (elapsed / 60 + 1e-9)
            eta = (total - done_so_far) / (rate + 1e-9)
            logger.info(
                f"[{done_so_far}/{total}] processed={processed} failed={failed} "
                f"| {rate:.1f} utt/min | ETA {eta:.1f} min"
            )

    executor.shutdown(wait=True)
    capture.remove_all()
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    logger.info(f"Done. {processed} processed, {skipped} skipped, {failed} failed in {elapsed/60:.1f} min")
    du = os.popen(f"du -sh {out_dir}").read().strip()
    logger.info(f"Disk usage: {du}")

    return {"family": family, "processed": processed, "skipped": skipped, "failed": failed}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True,
                   choices=["whisper_tiny", "whisper_small", "wavtok_40_unify",
                            "dacvae_stage1", "dacvae_stage2"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--speakers", default=None)
    p.add_argument("--max_utt_per_speaker", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    speakers = set(args.speakers.split(",")) if args.speakers else None

    result = extract_family(
        family=args.family,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        speakers=speakers,
        max_utt_per_speaker=args.max_utt_per_speaker,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
