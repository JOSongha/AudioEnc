"""Whisper-specific feature extraction helpers used by the omni dataset pipeline.

Decoupled from the dataset processor so the small numerical functions can be unit-tested
without HF dataset/tokenizer machinery.
"""

import math

import torch
from transformers import WhisperFeatureExtractor

# Encoder-output frame stride at 16 kHz.
# Whisper-small: mel hop = 10 ms (= 160 samples @ 16k); encoder Conv1d adds stride 2 →
# 1 encoder frame = 320 audio samples = 20 ms. 30 s × 50 fps = 1500 max frames.
WHISPER_HOP_LENGTH = 320
WHISPER_MAX_FRAMES = 1500
WHISPER_SAMPLE_RATE = 16000
WHISPER_MEL_BINS = 80
WHISPER_MEL_FRAMES = 3000  # 30 s × 100 fps mel
WHISPER_MAX_SAMPLES = 30 * WHISPER_SAMPLE_RATE  # 480_000

# Lazily-cached FeatureExtractor (one per process).  WhisperFeatureExtractor is pure-Python
# numpy under the hood; loading once amortizes the mel filterbank precompute.
_FE_CACHE: dict[str, WhisperFeatureExtractor] = {}


def get_feature_extractor(model_id: str = "openai/whisper-small.en") -> WhisperFeatureExtractor:
    if model_id not in _FE_CACHE:
        _FE_CACHE[model_id] = WhisperFeatureExtractor.from_pretrained(model_id)
    return _FE_CACHE[model_id]


def extract_mel(waveform: torch.Tensor, model_id: str = "openai/whisper-small.en") -> torch.Tensor:
    """Convert a 1-D mono waveform (float32, 16 kHz, ≤30 s) into [80, 3000] log-mel.

    The FeatureExtractor zero-pads short clips to 30 s and trims long ones; we still
    enforce ≤30 s explicitly to make the assumption visible at this layer.
    """
    if waveform.dim() == 2:
        if waveform.shape[0] != 1:
            raise ValueError(f"expected mono [1, S]; got {tuple(waveform.shape)}")
        waveform = waveform.squeeze(0)
    if waveform.dim() != 1:
        raise ValueError(f"expected 1-D waveform; got {tuple(waveform.shape)}")
    if waveform.numel() > WHISPER_MAX_SAMPLES:
        waveform = waveform[:WHISPER_MAX_SAMPLES]

    fe = get_feature_extractor(model_id)
    mel = fe(waveform.numpy(), sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt").input_features
    # mel shape: [1, 80, 3000] fp32
    return mel.squeeze(0)


def audio_pad_token_count(num_samples: int) -> int:
    """How many audio_pad_token slots correspond to a given waveform length.

    Mirrors Whisper encoder's downsampling: 1 encoder frame per 320 samples at 16 kHz, capped
    at 1500 (the 30-second hard limit). The runtime should slice encoder output to this many
    frames before feeding into the projector.
    """
    if num_samples <= 0:
        return 0
    return min(WHISPER_MAX_FRAMES, math.ceil(num_samples / WHISPER_HOP_LENGTH))
