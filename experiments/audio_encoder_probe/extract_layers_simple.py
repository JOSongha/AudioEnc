"""
Simple layer-wise extraction: encoder only.

For each family (whisper_tiny, whisper_small, dacvae, wavtok):
  - Forward through encoder with output_hidden_states=True (or hooks)
  - Extract 5 layer embeddings (sampled uniformly)
  - Pool: mean over valid frames
  - Save per-utterance npz

Usage:
  python extract_layers_simple.py --family whisper_tiny
  python extract_layers_simple.py --family whisper_small
  python extract_layers_simple.py --family dacvae
  python extract_layers_simple.py --family wavtok
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from experiments.audio_encoder_probe.extract_encoder_only import (
    ENCODER_CFG, AudioDataset, _collate, load_encoder, t_audio_for,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@torch.no_grad()
def extract_encoder_layers(
    encoder,
    encoder_type: str,
    audio_batch: torch.Tensor,
    t_audios: list[int],
    device: str,
) -> list[dict]:
    """
    Forward through encoder and extract 5 layer embeddings.

    Returns list of dicts per utterance with keys like enc_{0..4}_mean.
    """
    audio_batch = audio_batch.to(device)
    B = len(t_audios)

    if encoder_type in ("whisper_tiny", "whisper_small"):
        # Whisper: output_hidden_states=True gives 5 hidden states (embedding + 4 layers)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = encoder(audio_batch, output_hidden_states=True)

        hidden_states = out.hidden_states  # tuple of 5, each (B, T, D)
        all_hiddens = hidden_states  # Use all 5

    elif encoder_type == "dacvae":
        # DACVAE: use final output
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z = encoder.encode(audio_batch)  # (B, 128, T)

        if isinstance(z, tuple):
            z = z[0]

        # Transpose to (B, T, 128)
        enc_output = z.transpose(1, 2)
        # For 5 layers: replicate (placeholder for intermediate extraction)
        all_hiddens = [enc_output] * 5

    elif encoder_type == "wavtok":
        # WavTokenizer: use final output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder(audio_batch.to(torch.bfloat16))  # (B, D, T)

        if isinstance(z, tuple):
            z = z[0]

        # Transpose to (B, T, D)
        enc_output = z.transpose(1, 2)
        # For 5 layers: replicate
        all_hiddens = [enc_output] * 5

    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    # Pool embeddings per utterance
    results = []
    for b in range(B):
        result = {}

        for layer_idx, hidden in enumerate(all_hiddens):
            h = hidden[b, :t_audios[b]].float()  # (t, D)
            mean_pool = h.mean(dim=0).cpu().numpy().astype(np.float32)
            result[f'enc_{layer_idx}_mean'] = mean_pool

        results.append(result)

    return results


def main(family: str, manifest_path: str = None, device: str = "cuda", batch_size: int = 1):
    """Extract encoder layer embeddings for a family."""
    from transformers import WhisperModel, AutoModelForCausalLM

    root = Path(__file__).resolve().parent.parent.parent

    if manifest_path is None:
        manifest_path = root / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"
    else:
        manifest_path = root / manifest_path

    out_dir = root / f"experiments/audio_encoder_probe/embeds_layers/{family}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(df)} utterances")

    logger.info(f"Loading encoder for family: {family}")
    cfg_key = "wavtok_40_unify" if family == "wavtok" else family
    cfg = ENCODER_CFG[cfg_key]

    if family in ("whisper_tiny", "whisper_small"):
        # Load Whisper encoder
        whisper = WhisperModel.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16)
        encoder = whisper.encoder.to(device).eval()
        del whisper
    elif family == "dacvae":
        base = str(cfg["base_model_dir"])
        alm = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
            device_map=device,
        )
        encoder = alm.model.audio_encoder.encoder.eval()
    elif family == "wavtok":
        base = str(cfg["base_model_dir"])
        alm = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
            device_map=device,
        )
        encoder = alm.model.audio_encoder.encoder.eval()
    else:
        raise ValueError(f"Unknown family: {family}")

    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("Encoder loaded successfully")

    logger.info("Processing utterances...")

    # Create dataset
    rows = df.to_dict('records')
    dataset = AudioDataset(rows, family)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, num_workers=0)

    processed = 0
    for batch_idx, batch in enumerate(loader):
        valid_rows = batch['valid']
        errors = batch['errors']

        if errors:
            for utt_id, err in errors:
                logger.warning(f"Skipped {utt_id}: {err}")

        if not valid_rows:
            continue

        utt_ids = batch['utt_ids']
        audio_batch = batch['audio_batch']
        t_audios = batch['t_audios']

        results = extract_encoder_layers(
            encoder=encoder,
            encoder_type=family,
            audio_batch=audio_batch,
            t_audios=t_audios,
            device=device,
        )

        # Save per-utterance npz
        for utt_id, result in zip(utt_ids, results):
            npz_path = out_dir / f"{utt_id}.npz"
            np.savez(str(npz_path), **result)
            processed += 1

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(df)} utterances")

    logger.info(f"✓ Completed! Saved {processed} utterances to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["whisper_tiny", "whisper_small", "dacvae", "wavtok"])
    parser.add_argument("--manifest", default="experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(args.family, args.manifest, args.device, args.batch_size)
