#!/usr/bin/env python3
"""
Build CSV manifest for CMU-ARCTIC dataset.

This script loads transcripts from the HuggingFace MikhailT/cmu-arctic dataset
and matches them with local WAV files to generate a CSV manifest with the following columns:
  - audio_path: absolute path to WAV file
  - speaker_id: speaker folder name (bdl, slt, jmk, awb, rms, clb, ksp, and others)
  - utt_id: utterance ID derived from filename (e.g., bdl-a0001)
  - transcript_id: same as utt_id (CMU-ARCTIC convention)
  - transcript: transcript text from HuggingFace dataset
  - duration_s: duration in seconds

The manifest is saved to /mnt/tmp/cache/cmu_arctic_7/manifest.csv
"""

import os
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

DATASET_ROOT = Path("/mnt/tmp/cache/cmu_arctic_7")
SPEAKERS = ["bdl", "slt", "jmk", "awb", "rms", "clb", "ksp"]


def load_transcripts_from_hf() -> Dict[str, Dict[str, str]]:
    """Load transcripts from HuggingFace MikhailT/cmu-arctic dataset."""
    print("Loading transcripts from HuggingFace dataset...")
    try:
        ds = load_dataset("MikhailT/cmu-arctic", cache_dir="/mnt/tmp/cache/hf")
        transcripts = {}

        # Each split is a speaker (aew, ahw, etc.)
        for speaker_id in ds.keys():
            split_data = ds[speaker_id]
            speaker_transcripts = {}

            # Build mapping of file->text
            for idx in range(len(split_data)):
                file_name = split_data['file'][idx]  # e.g., "cmu_us_aew_arctic_a0001.wav"
                text = split_data['text'][idx]

                # Extract utt_id from file name
                # File format: cmu_us_{speaker}_{utt_id}.wav -> utt_id
                if file_name.startswith("cmu_us_"):
                    # Remove prefix and .wav extension
                    parts = file_name.replace("cmu_us_", "").replace(".wav", "").split("_", 1)
                    if len(parts) == 2:
                        utt_id = f"{parts[0]}-{parts[1]}"
                    else:
                        utt_id = file_name.replace(".wav", "")
                else:
                    utt_id = file_name.replace(".wav", "")

                speaker_transcripts[utt_id] = text

            transcripts[speaker_id] = speaker_transcripts
            print(f"  Loaded {speaker_id}: {len(speaker_transcripts)} transcripts")

        return transcripts
    except Exception as e:
        print(f"Error loading transcripts from HuggingFace: {e}")
        return {}


def get_duration(audio_path: str) -> float:
    """Get duration in seconds from audio file."""
    try:
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception as e:
        print(f"Warning: Failed to get duration for {audio_path}: {e}")
        return 0.0


def build_manifest() -> pd.DataFrame:
    """Build manifest dataframe from dataset structure."""
    records: List[Dict[str, Any]] = []
    transcripts = load_transcripts_from_hf()

    for speaker_id in SPEAKERS:
        speaker_dir = DATASET_ROOT / speaker_id

        if not speaker_dir.exists():
            print(f"Warning: Speaker directory not found: {speaker_dir}")
            continue

        # Find all WAV files in speaker directory
        wav_files = sorted(speaker_dir.glob("*.wav"))

        for wav_file in wav_files:
            audio_path = str(wav_file.resolve())

            # Extract utt_id from filename (remove .wav extension)
            # Files are named like: bdl-a0001.wav
            utt_id = wav_file.stem
            transcript_id = utt_id

            # Try to get transcript from loaded transcripts
            transcript = "TRANSCRIPT_MISSING"
            if speaker_id in transcripts and utt_id in transcripts[speaker_id]:
                transcript = transcripts[speaker_id][utt_id]

            # Get duration
            duration_s = get_duration(audio_path)

            records.append({
                "audio_path": audio_path,
                "speaker_id": speaker_id,
                "utt_id": utt_id,
                "transcript_id": transcript_id,
                "transcript": transcript,
                "duration_s": duration_s,
            })

        print(f"Processed {speaker_id}: {len(wav_files)} utterances")

    df = pd.DataFrame(records)
    return df


def verify_manifest(df: pd.DataFrame) -> None:
    """Verify manifest quality."""
    print("\n" + "="*80)
    print("MANIFEST VERIFICATION")
    print("="*80)

    print(f"\nManifest shape: {df.shape}")
    print(f"Total utterances: {len(df)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSpeaker statistics:")
    speaker_counts = df["speaker_id"].value_counts().sort_index()
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count} utterances")

    print(f"\nDuration statistics:")
    print(f"  Min: {df['duration_s'].min():.2f}s")
    print(f"  Max: {df['duration_s'].max():.2f}s")
    print(f"  Mean: {df['duration_s'].mean():.2f}s")
    print(f"  Median: {df['duration_s'].median():.2f}s")

    # Check for missing transcripts
    missing_transcripts = (df["transcript"] == "TRANSCRIPT_MISSING").sum()
    print(f"\nTranscripts with TRANSCRIPT_MISSING: {missing_transcripts}")

    # Verify all columns
    print(f"\nColumns: {list(df.columns)}")
    print(f"Null values per column:\n{df.isnull().sum()}")

    print("\n" + "="*80)


def main() -> None:
    """Build and save manifest."""
    print("Building CMU-ARCTIC manifest...")

    df = build_manifest()

    # Verify manifest
    verify_manifest(df)

    # Save to CSV
    output_path = DATASET_ROOT / "manifest.csv"
    df.to_csv(output_path, index=False)
    print(f"\nManifest saved to: {output_path}")


if __name__ == "__main__":
    main()
