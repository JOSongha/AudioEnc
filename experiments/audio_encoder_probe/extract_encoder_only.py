"""
Audio-encoder-only embedding extraction.

For each (encoder_type, dataset) pair:
  - Load audio at encoder's required sample rate
  - Forward through encoder ONLY (no projector, no LLM)
  - Mean-pool over valid frames
  - Save (d,) float32 .npy per utterance

Encoders:
  whisper_small : openai/whisper-small.en encoder (768-d, 50 fps, 16 kHz)
  dacvae        : Stage1 ckpt's audio_encoder.encoder (DACVAE z_e: 128-d, 25 fps, 48 kHz)

Usage:
  python extract_encoder_only.py --encoder whisper_small --manifest manifests/iemocap_4class.csv \
      --out_dir experiments/audio_encoder_probe/whisper_small/iemocap [--batch_size 16] [--num_workers 4]
"""

import argparse
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Encoder configs ──────────────────────────────────────────────────────────

ENCODER_CFG = {
    "mel_only": {
        # Baseline: mean over valid log-mel frames (no encoder forward)
        "sample_rate": 16000,
        "hop_length": 160,    # mel hop (10 ms); 100 fps
        "max_frames": 3000,   # 30 s at 100 fps
        "max_samples": 30 * 16000,
        "model_id": "openai/whisper-small.en",  # any whisper FE works
        "embed_dim": 80,
    },
    "whisper_tiny": {
        "sample_rate": 16000,
        "hop_length": 320,
        "max_frames": 1500,
        "max_samples": 30 * 16000,
        "model_id": "openai/whisper-tiny.en",
        "embed_dim": 384,
    },
    "whisper_small": {
        "sample_rate": 16000,
        "hop_length": 320,
        "max_frames": 1500,
        "max_samples": 30 * 16000,
        "model_id": "openai/whisper-small.en",
        "embed_dim": 768,
    },
    "dacvae": {
        "sample_rate": 48000,
        "hop_length": 1920,
        "max_frames": None,
        "max_samples": None,
        "base_model_dir": REPO / "external/ckpts/Qwen3.5AE-4B-dacvae_ASR-Stage1",
        "embed_dim": 128,
    },
    "wavtok_40_unify": {
        "sample_rate": 24000,
        "hop_length": 600,    # 24kHz / 40fps; ratios [6,5,5,4]
        "max_frames": None,
        "max_samples": None,
        "base_model_dir": REPO / "external/models/Qwen3.5AE-4B-wavtok-40-unify",
        "embed_dim": 512,
    },
    "wavtok_40_unify_postvq": {
        # Post-VQ: encoder forward, then nearest-neighbor lookup in WavTokenizer codebook
        "sample_rate": 24000,
        "hop_length": 600,
        "max_frames": None,
        "max_samples": None,
        "base_model_dir": REPO / "external/models/Qwen3.5AE-4B-wavtok-40-unify",
        "wavtok_ckpt_hf_repo": "novateur/WavTokenizer-large-unify-40token",
        "wavtok_ckpt_filename": "wavtokenizer_large_unify_600_24k.ckpt",
        "embed_dim": 512,
    },
    "encodec_24k": {
        "sample_rate": 24000,
        "hop_length": 320,    # 24kHz / 75fps; upsampling_ratios [8,5,4,2] product
        "max_frames": None,
        "max_samples": None,
        "base_model_dir": REPO / "external/models/Qwen3.5AE-4B-encodec-24k",
        "embed_dim": 128,
    },
}


def t_audio_for(encoder_type: str, num_samples: int) -> int:
    cfg = ENCODER_CFG[encoder_type]
    if cfg["max_frames"] is not None:
        return min(cfg["max_frames"], math.ceil(num_samples / cfg["hop_length"]))
    return num_samples // cfg["hop_length"]


# ── Dataset ──────────────────────────────────────────────────────────────────

_whisper_fe_cache = {}

def _get_whisper_fe(model_id: str):
    if model_id not in _whisper_fe_cache:
        from transformers import WhisperFeatureExtractor
        _whisper_fe_cache[model_id] = WhisperFeatureExtractor.from_pretrained(model_id)
    return _whisper_fe_cache[model_id]


class AudioDataset(Dataset):
    def __init__(self, rows, encoder_type: str):
        self.rows = rows
        self.encoder_type = encoder_type
        self.cfg = ENCODER_CFG[encoder_type]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        utt_id = str(row["utt_id"])
        path = str(row["audio_path"])

        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            return {"utt_id": utt_id, "error": f"load:{e}"}

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_sr = self.cfg["sample_rate"]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        # Truncate if too long (Whisper 30s)
        if self.cfg["max_samples"] is not None and wav.shape[-1] > self.cfg["max_samples"]:
            wav = wav[:, :self.cfg["max_samples"]]
        n_samples = wav.shape[-1]
        t_audio = t_audio_for(self.encoder_type, n_samples)
        if t_audio == 0:
            return {"utt_id": utt_id, "error": "t_audio=0"}

        if self.encoder_type in ("whisper_small", "whisper_tiny", "mel_only"):
            fe = _get_whisper_fe(self.cfg["model_id"])
            mel = fe(wav.squeeze(0).numpy(), sampling_rate=target_sr,
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


def _collate(batch):
    valid = [b for b in batch if "error" not in b]
    errors = [(b["utt_id"], b["error"]) for b in batch if "error" in b]
    if not valid:
        return {"valid": [], "errors": errors}

    tensors = [b["audio_tensor"] for b in valid]
    if tensors[0].ndim == 2 and tensors[0].shape[0] == 80:
        # Whisper mel — stack
        audio_batch = torch.stack(tensors, dim=0)  # (B, 80, 3000)
    else:
        # waveform — pad to max length
        max_s = max(t.shape[-1] for t in tensors)
        audio_batch = torch.stack(
            [F.pad(t, (0, max_s - t.shape[-1])) for t in tensors], dim=0
        )  # (B, 1, max_s)

    return {
        "valid": valid,
        "utt_ids": [b["utt_id"] for b in valid],
        "t_audios": [b["t_audio"] for b in valid],
        "audio_batch": audio_batch,
        "errors": errors,
    }


# ── Encoder loading ──────────────────────────────────────────────────────────

def load_encoder(encoder_type: str, device: str, dtype=torch.bfloat16):
    cfg = ENCODER_CFG[encoder_type]
    if encoder_type == "mel_only":
        # No encoder; we'll mean-pool the mel directly in encode_batch
        return None
    if encoder_type in ("whisper_small", "whisper_tiny"):
        from transformers import WhisperModel
        whisper = WhisperModel.from_pretrained(
            cfg["model_id"],
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        encoder = whisper.encoder.to(device).eval()
        del whisper
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    if encoder_type == "wavtok_40_unify_postvq":
        from transformers import AutoModelForCausalLM
        from huggingface_hub import hf_hub_download

        base = str(cfg["base_model_dir"])
        logger.info(f"Loading wavtok base for {encoder_type}: {base}")
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(device)
        encoder = model.model.audio_encoder.encoder
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        # Load codebook from WavTokenizer ckpt
        ckpt_local = hf_hub_download(
            repo_id=cfg["wavtok_ckpt_hf_repo"],
            filename=cfg["wavtok_ckpt_filename"],
            repo_type="model",
        )
        ck = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        codebook_key = "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed"
        codebook = ck["state_dict"][codebook_key].float().to(device)  # (4096, 512)
        logger.info(f"Loaded codebook: {tuple(codebook.shape)}")

        # Attach codebook to encoder for retrieval in encode_batch
        encoder._codebook = codebook  # (4096, 512) float32 on device

        # Drop rest of model
        del model.model.audio_encoder.projector
        del model.model.language_model
        del model.lm_head
        torch.cuda.empty_cache()
        return encoder

    if encoder_type in ("dacvae", "wavtok_40_unify", "encodec_24k"):
        from transformers import AutoModelForCausalLM
        base = str(cfg["base_model_dir"])
        logger.info(f"Loading model for {encoder_type} encoder: {base}")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(device)
        encoder = model.model.audio_encoder.encoder
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        # Drop the rest of the model
        del model.model.audio_encoder.projector
        del model.model.language_model
        del model.lm_head
        torch.cuda.empty_cache()
        return encoder

    raise ValueError(f"Unknown encoder: {encoder_type}")


# ── Forward + pool ───────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(encoder, encoder_type: str, audio_batch: torch.Tensor,
                 t_audios: list[int], device: str) -> list[np.ndarray]:
    """
    Returns list of (d,) float32 numpy arrays, one per sample.
    """
    audio_batch = audio_batch.to(device)

    if encoder_type == "mel_only":
        # audio_batch is mel (B, 80, 3000); pool over valid mel frames.
        out_list = []
        for b, t in enumerate(t_audios):
            mel_b = audio_batch[b].float().cpu()  # (80, 3000)
            valid = mel_b[:, :t]                  # (80, t)
            if valid.shape[1] == 0:
                out_list.append(None)
                continue
            out_list.append(valid.mean(dim=-1).numpy().astype(np.float32))
        return out_list

    if encoder_type in ("whisper_small", "whisper_tiny"):
        # WhisperEncoder expects (B, 80, 3000); returns BaseModelOutput
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = encoder(audio_batch)
        h = out.last_hidden_state  # (B, 1500, d)
    elif encoder_type == "dacvae":
        # DACVAE.encode expects (B, 1, S); returns z_e: (B, D, T)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z = encoder.encode(audio_batch)
        if isinstance(z, tuple):
            z = z[0]
        h = z.transpose(1, 2)  # (B, T, D)
    elif encoder_type in ("wavtok_40_unify", "encodec_24k"):
        # SEANet / EnCodec encoder.forward expects (B, 1, S); returns (B, D, T)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder(audio_batch.to(torch.bfloat16))
        if isinstance(z, tuple):
            z = z[0]
        h = z.transpose(1, 2)  # (B, T, D)
    elif encoder_type == "wavtok_40_unify_postvq":
        # Encoder forward → z_e: (B, 512, T) → (B, T, 512)
        # Then nearest-neighbor codebook lookup → z_q: (B, T, 512)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z_e = encoder(audio_batch.to(torch.bfloat16))  # (B, 512, T)
        z_e = z_e.float().transpose(1, 2)  # (B, T, 512)
        codebook = encoder._codebook                       # (4096, 512) float32
        # squared distance: (B, T, 4096) — we use ||z||² + ||c||² - 2 z·c
        # but only argmin over c, so ||z||² constant per (b,t) drops out.
        # argmin ||z - c||² = argmax(2 z·c - ||c||²)
        cb_norm_sq = (codebook ** 2).sum(dim=-1)           # (4096,)
        # (B, T, 512) @ (512, 4096) → (B, T, 4096)
        scores = z_e @ codebook.T * 2 - cb_norm_sq[None, None, :]
        idx = scores.argmax(dim=-1)                        # (B, T)
        z_q = codebook[idx]                                # (B, T, 512)
        h = z_q  # (B, T, D)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    h = h.float().cpu()  # (B, T, D)
    out_list = []
    for b, t in enumerate(t_audios):
        valid = h[b, :t]   # (t, D)
        if valid.shape[0] == 0:
            out_list.append(None)
            continue
        out_list.append(valid.mean(dim=0).numpy().astype(np.float32))
    return out_list


# ── Main loop ────────────────────────────────────────────────────────────────

def extract(encoder_type: str, manifest_path: str, out_dir: str,
            batch_size: int = 16, num_workers: int = 4, device: str = "cuda:0"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    rows = [r for r in df.to_dict("records") if not (out_dir / f"{r['utt_id']}.npy").exists()]
    logger.info(f"To process: {len(rows)} (skip {len(df) - len(rows)} already done)")

    if not rows:
        logger.info("Nothing to do.")
        return

    encoder = load_encoder(encoder_type, device=device)
    logger.info(f"Encoder loaded: {encoder.__class__.__name__}")

    dataset = AudioDataset(rows, encoder_type)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )

    executor = ThreadPoolExecutor(max_workers=4)

    def _save(utt_id, vec):
        np.save(str(out_dir / f"{utt_id}.npy"), vec)

    processed = failed = 0
    import time
    start = time.time()
    total = len(rows)

    for batch_idx, batch in enumerate(loader):
        for uid, err in batch.get("errors", []):
            logger.warning(f"err {uid}: {err}")
            failed += 1
        if not batch.get("valid"):
            continue

        try:
            embeds = encode_batch(encoder, encoder_type,
                                  batch["audio_batch"], batch["t_audios"], device)
        except Exception as e:
            logger.warning(f"Batch encode failed: {e}")
            failed += len(batch["utt_ids"])
            continue

        for uid, vec in zip(batch["utt_ids"], embeds):
            if vec is None:
                failed += 1
                continue
            executor.submit(_save, uid, vec)
            processed += 1

        done = processed + failed
        if (batch_idx + 1) % 20 == 0 or done >= total:
            elapsed = time.time() - start
            rate = processed / (elapsed / 60 + 1e-9)
            eta = (total - done) / (rate + 1e-9)
            logger.info(f"[{done}/{total}] proc={processed} fail={failed} | {rate:.1f} utt/min | ETA {eta:.1f} min")

    executor.shutdown(wait=True)
    elapsed = time.time() - start
    logger.info(f"Done. {processed} processed, {failed} failed in {elapsed/60:.1f} min")
    logger.info(f"Output: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODER_CFG.keys()))
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    extract(args.encoder, args.manifest, args.out_dir,
            args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()
