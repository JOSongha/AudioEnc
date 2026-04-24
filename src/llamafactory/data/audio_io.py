"""Audio I/O helpers: file load with optional time-range slicing and resampling.

Used by Whisper-based omni dataset to support manifest entries with `audio_start`/`audio_end`
(produced by `scripts/split_long_audio_30s.py`).
"""

import io
import math

import torch
import torchaudio


def load_audio_chunk(
    source,
    target_sr: int = 16000,
    audio_start: float | None = None,
    audio_end: float | None = None,
) -> torch.Tensor:
    """Load audio file (or bytes/BytesIO), optionally slice [audio_start, audio_end] seconds,
    resample to target_sr, mono-mix.

    Returns a 1-D tensor [S] at target_sr (mono, fp32). Raises on load failure.

    Slicing happens BEFORE resample, using `frame_offset`/`num_frames` so we don't read the
    whole file when only a chunk is needed.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    if audio_start is None and audio_end is None:
        wav, sr = torchaudio.load(source)
    else:
        # Need original sample rate to compute frame offsets
        info = torchaudio.info(source)
        sr_orig = info.sample_rate
        start_sec = audio_start or 0.0
        frame_offset = int(round(start_sec * sr_orig))
        if audio_end is None:
            num_frames = -1
        else:
            num_frames = max(0, int(round((audio_end - start_sec) * sr_orig)))
        if isinstance(source, io.BytesIO):
            source.seek(0)
        wav, sr = torchaudio.load(source, frame_offset=frame_offset, num_frames=num_frames)

    # Mono mix
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0).to(torch.float32)
