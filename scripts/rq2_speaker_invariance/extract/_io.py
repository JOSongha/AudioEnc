"""Shared helpers for the RQ2 extractors."""

import json
from pathlib import Path

import numpy as np
import torch


def load_pairs(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_wav(path: str, target_sr: int = 16000, max_sec: float = 30.0):
    """Return float32 mono waveform sampled at target_sr, clipped to max_sec."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import resampy
        wav = resampy.resample(wav, sr, target_sr).astype(np.float32)
    if max_sec is not None:
        max_samples = int(max_sec * target_sr)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
    return wav


def mean_pool(hidden: torch.Tensor, valid_len: int | None = None) -> torch.Tensor:
    """`hidden`: [T, D]. Returns [D] = mean over T (or first `valid_len` frames)."""
    if valid_len is not None:
        hidden = hidden[:valid_len]
    return hidden.float().mean(dim=0)


def save_layer_npz(out_dir: Path, model_tag: str, layer_tag: str, rows: list[dict]):
    """rows: list of {emb: np.ndarray[D], pair, spk, src}. Writes one npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = np.stack([r["emb"] for r in rows]).astype(np.float32)
    pair = np.array([r["pair"] for r in rows])
    spk = np.array([r["spk"] for r in rows])
    src = np.array([r["src"] for r in rows])
    out_path = out_dir / f"{model_tag}__{layer_tag}.npz"
    np.savez_compressed(
        out_path,
        emb=embs,
        pair=pair,
        spk=spk,
        src=src,
        model=model_tag,
        layer=layer_tag,
    )
    return out_path
