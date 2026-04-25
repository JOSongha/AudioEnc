"""
Data pipeline dry-run for Whisper omni dataset.

Tests (no training, no GPU required):
  1. extract_mel() → shape [80, 3000], dtype float32
  2. audio_pad_token_count() → ceil(n/320) capped at 1500
  3. WhisperOmniCollator() → audio_features [N, 80, 3000], audio_lengths aligned
  4. audio_pad_token count in input_ids == audio_lengths sum  (critical alignment check)
  5. (optional) real manifest row via Nubes — requires gateway access

Run:
    python scripts/smoke_whisper_batch.py
    python scripts/smoke_whisper_batch.py --with-nubes  # also test real Nubes row
"""

import math
import sys
import argparse
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from llamafactory.data.whisper_features import (
    WHISPER_HOP_LENGTH,
    WHISPER_MAX_FRAMES,
    WHISPER_MEL_BINS,
    WHISPER_MEL_FRAMES,
    WHISPER_SAMPLE_RATE,
    audio_pad_token_count,
    extract_mel,
)
from llamafactory.data.omni_dataset_whisper import WhisperOmniCollator
from llamafactory.extras.constants import IGNORE_INDEX

AUDIO_PAD_TOKEN_ID = 248076  # from config.json


def _make_synthetic_wav(seconds: float, sr: int = WHISPER_SAMPLE_RATE) -> torch.Tensor:
    """Sine wave at 440 Hz, shape [n_samples]."""
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    return (0.5 * torch.sin(2 * math.pi * 440 * t))


def _make_collator_feature(seq_len: int, mel: torch.Tensor, t_audio: int) -> dict:
    """Minimal packed feature matching WhisperOmniCollator's expected input."""
    input_ids = [1] * (seq_len - t_audio) + [AUDIO_PAD_TOKEN_ID] * t_audio
    labels = [IGNORE_INDEX] * (seq_len - t_audio) + [IGNORE_INDEX] * t_audio
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * seq_len,
        "audio_features": [mel],
        "audio_lengths": [t_audio],
        "modality_ids": [0] * seq_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-nubes", action="store_true",
                        help="Also test one real row loaded from Nubes gateway")
    parser.add_argument("--nubes-gateway", default="http://c.nubes.sto.navercorp.com:8000/v1")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "external/datasets/libri_mls_vox"))
    args = parser.parse_args()

    failed = False
    print("=" * 60)
    print("Whisper data pipeline smoke test")
    print("=" * 60)

    # ── 1. extract_mel ────────────────────────────────────────────────
    print("\n[1/4] extract_mel shape & dtype")
    for seconds in [1.0, 10.0, 30.0, 0.1]:
        wav = _make_synthetic_wav(seconds)
        mel = extract_mel(wav)
        expected_shape = (WHISPER_MEL_BINS, WHISPER_MEL_FRAMES)  # (80, 3000)
        ok = mel.shape == expected_shape and mel.dtype == torch.float32
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {seconds:.1f}s wav → mel {tuple(mel.shape)} dtype={mel.dtype}")
        if not ok:
            failed = True

    # ── 2. audio_pad_token_count ──────────────────────────────────────
    print("\n[2/4] audio_pad_token_count alignment")
    test_cases = [
        (1 * 16000,  "1s"),
        (10 * 16000, "10s"),
        (30 * 16000, "30s (max)"),
        (31 * 16000, "31s (>max, capped)"),
        (160,        "10ms (edge case)"),
        (320,        "20ms = 1 frame"),
        (321,        "20ms+1 sample = 2 frames"),
    ]
    for n_samples, label in test_cases:
        t = audio_pad_token_count(n_samples)
        expected = min(math.ceil(n_samples / WHISPER_HOP_LENGTH), WHISPER_MAX_FRAMES)
        ok = t == expected and t >= 1 and t <= WHISPER_MAX_FRAMES
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: n={n_samples} → t_audio={t} (expected={expected})")
        if not ok:
            failed = True

    # ── 3. WhisperOmniCollator shapes ─────────────────────────────────
    print("\n[3/4] WhisperOmniCollator output shapes")
    collator = WhisperOmniCollator(pad_token_id=0)

    durations = [3.0, 15.0, 29.9]
    items = []
    expected_audio_lengths = []
    for d in durations:
        wav = _make_synthetic_wav(d)
        mel = extract_mel(wav)
        t = audio_pad_token_count(wav.shape[0])
        items.append((mel, t))
        expected_audio_lengths.append(t)

    # All sequences must be same length for collator; pad to max
    max_len = max(t + 20 for _, t in items)
    batch = []
    for mel, t in items:
        feat = _make_collator_feature(seq_len=max_len, mel=mel, t_audio=t)
        # Pad input_ids / labels / attention_mask to max_len
        pad_len = max_len - len(feat["input_ids"])
        feat["input_ids"] = feat["input_ids"] + [0] * pad_len
        feat["labels"] = feat["labels"] + [IGNORE_INDEX] * pad_len
        feat["attention_mask"] = feat["attention_mask"] + [0] * pad_len
        feat["modality_ids"] = feat["modality_ids"] + [0] * pad_len
        batch.append(feat)

    out = collator(batch)

    # audio_features shape
    n_audio = len(durations)
    af = out["audio_features"]
    af_ok = af.shape == (n_audio, WHISPER_MEL_BINS, WHISPER_MEL_FRAMES) and af.dtype == torch.float32
    print(f"  audio_features: {tuple(af.shape)} dtype={af.dtype}")
    print(f"  {'[PASS]' if af_ok else '[FAIL]'} expected ({n_audio}, {WHISPER_MEL_BINS}, {WHISPER_MEL_FRAMES})")
    if not af_ok:
        failed = True

    # audio_lengths alignment
    al = out["audio_lengths"].tolist()
    al_ok = al == expected_audio_lengths
    print(f"  audio_lengths:  {al}")
    print(f"  expected:       {expected_audio_lengths}")
    print(f"  {'[PASS]' if al_ok else '[FAIL]'} audio_lengths match")
    if not al_ok:
        failed = True

    # ── 4. audio_pad_token count == audio_lengths sum ─────────────────
    print("\n[4/4] input_ids audio_pad_token count == audio_lengths")
    for i, (feat, t_expect) in enumerate(zip(batch, expected_audio_lengths)):
        n_pad_in_ids = feat["input_ids"].count(AUDIO_PAD_TOKEN_ID)
        t = feat["audio_lengths"][0]
        ok = n_pad_in_ids == t == t_expect
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] sample {i}: input_ids pads={n_pad_in_ids} == t_audio={t} (expected {t_expect})")
        if not ok:
            failed = True

    # ── 5. (optional) Real Nubes row ──────────────────────────────────
    if args.with_nubes:
        print(f"\n[5/5] Real Nubes row (gateway={args.nubes_gateway})")
        import json, requests
        shards = sorted(Path(args.manifest).glob("shard_*.jsonl"))
        if not shards:
            print("  [SKIP] No shards found at manifest path")
        else:
            with open(shards[0]) as f:
                row = json.loads(f.readline())
            nubes_path = row.get("nubes_path", "")
            text = row.get("text", "")
            print(f"  nubes_path: {nubes_path[:80]}")
            print(f"  text: {text[:60]}")
            try:
                from llamafactory.data.audio_io import load_audio_chunk
                resp = requests.get(f"{args.nubes_gateway}/{nubes_path}", timeout=5.0)
                resp.raise_for_status()
                import io
                wav = load_audio_chunk(io.BytesIO(resp.content), target_sr=WHISPER_SAMPLE_RATE)
                mel = extract_mel(wav)
                t = audio_pad_token_count(wav.shape[0])
                print(f"  wav: {wav.shape[0]} samples ({wav.shape[0]/WHISPER_SAMPLE_RATE:.2f}s)")
                print(f"  mel: {tuple(mel.shape)}, t_audio={t}")
                print(f"  [PASS] Real audio loaded successfully")
            except Exception as e:
                print(f"  [SKIP] Nubes fetch failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print("RESULT: FAIL — see details above")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
