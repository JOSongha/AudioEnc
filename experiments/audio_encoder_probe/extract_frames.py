"""
Frame-level (T, D) embedding extraction. Reuses extract_encoder_only.py's
audio loading + encoder forward, but saves raw frames instead of mean-pooled.

Output: .npz per utterance
  frames  (T, D) float16 — valid frames only, downsampled to ≤MAX_FRAMES uniformly
  n_valid int             — original valid frame count (before downsample)
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from experiments.audio_encoder_probe.extract_encoder_only import (
    AudioDataset, _collate, ENCODER_CFG, load_encoder
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_FRAMES = 512


@torch.no_grad()
def encode_batch_frames(encoder, encoder_type, audio_batch, t_audios, device,
                        max_frames=MAX_FRAMES):
    """Returns list of dict {frames: (T_ds, D) np.float16, n_valid: int} per sample."""
    audio_batch = audio_batch.to(device)

    if encoder_type == "mel_only":
        h = audio_batch.float().cpu().numpy()  # (B, 80, 3000)
        out = []
        for b, t in enumerate(t_audios):
            valid = h[b, :, :t].T  # (t, 80)
            if valid.shape[0] > max_frames:
                idx = np.round(np.linspace(0, valid.shape[0]-1, max_frames)).astype(int)
                valid = valid[idx]
            out.append({"frames": valid.astype(np.float16), "n_valid": int(t)})
        return out

    if encoder_type in ("whisper_small", "whisper_tiny"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            ho = encoder(audio_batch)
        h = ho.last_hidden_state.float()  # (B, 1500, D)
    elif encoder_type == "dacvae":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder.encode(audio_batch)
        if isinstance(z, tuple):
            z = z[0]
        h = z.float().transpose(1, 2)  # (B, T, D)
    elif encoder_type in ("wavtok_40_unify", "encodec_24k"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder(audio_batch.to(torch.bfloat16))
        if isinstance(z, tuple):
            z = z[0]
        h = z.float().transpose(1, 2)
    else:
        raise ValueError(encoder_type)

    h = h.cpu().numpy()  # (B, T_total, D)
    out = []
    for b, t in enumerate(t_audios):
        valid = h[b, :t]  # (t, D)
        if valid.shape[0] > max_frames:
            idx = np.round(np.linspace(0, valid.shape[0]-1, max_frames)).astype(int)
            valid = valid[idx]
        out.append({"frames": valid.astype(np.float16), "n_valid": int(t)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODER_CFG.keys()))
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    rows = [r for r in df.to_dict("records")
            if not (out_dir / f"{r['utt_id']}.npz").exists()]
    logger.info(f"To process: {len(rows)} (skip {len(df) - len(rows)} done)")

    if not rows:
        return

    encoder = load_encoder(args.encoder, device=args.device, dtype=torch.bfloat16)
    cfg = ENCODER_CFG[args.encoder]

    dataset = AudioDataset(rows, args.encoder)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=_collate, prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )

    executor = ThreadPoolExecutor(max_workers=4)

    def _save(utt_id, data):
        np.savez_compressed(str(out_dir / f"{utt_id}.npz"),
                            frames=data["frames"],
                            n_valid=np.array(data["n_valid"], dtype=np.int32))

    proc = fail = 0
    start = time.time()
    total = len(rows)
    for batch_idx, batch in enumerate(loader):
        for uid, err in batch.get("errors", []):
            logger.warning(f"err {uid}: {err}")
            fail += 1
        if not batch.get("valid"):
            continue
        try:
            frames = encode_batch_frames(encoder, args.encoder,
                                         batch["audio_batch"], batch["t_audios"],
                                         args.device)
        except Exception as e:
            logger.warning(f"batch fail: {e}")
            fail += len(batch["utt_ids"])
            continue

        for uid, data in zip(batch["utt_ids"], frames):
            executor.submit(_save, uid, data)
            proc += 1

        done = proc + fail
        if (batch_idx+1) % 10 == 0 or done >= total:
            elapsed = time.time() - start
            rate = proc / (elapsed/60 + 1e-9)
            eta = (total - done)/(rate + 1e-9)
            logger.info(f"[{done}/{total}] {rate:.0f} utt/min ETA {eta:.1f}m")

    executor.shutdown(wait=True)
    elapsed = time.time() - start
    logger.info(f"Done {proc} fail {fail} in {elapsed/60:.1f}m → {out_dir}")


if __name__ == "__main__":
    main()
