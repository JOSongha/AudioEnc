"""
Build balanced manifest: 2000 utterance (class당 500) from IEMOCAP 4-class.

Filters:
  - duration_s >= 1.0
  - Seed=42 for reproducibility

Outputs:
  iemocap_4class_balanced500.csv (2000 rows)
"""
import pandas as pd
from pathlib import Path


def build_balanced_manifest(input_csv: str, output_csv: str, n_per_class: int = 500, seed: int = 42, min_duration: float = 1.0):
    """
    Build balanced manifest by sampling n_per_class utterances from each emotion class.
    """
    df = pd.read_csv(input_csv)

    # Filter by minimum duration
    df = df[df['duration_s'] >= min_duration].reset_index(drop=True)
    print(f"After filtering (duration >= {min_duration}s): {len(df)} utterances")

    # Check class distribution
    print("\nClass distribution before sampling:")
    print(df['emotion'].value_counts().sort_index())

    # Stratified sampling: n_per_class per emotion
    balanced = []
    for emotion in sorted(df['emotion'].unique()):
        emotion_df = df[df['emotion'] == emotion]
        if len(emotion_df) < n_per_class:
            raise ValueError(
                f"Emotion '{emotion}' has only {len(emotion_df)} utterances "
                f"(need {n_per_class})"
            )
        sampled = emotion_df.sample(n=n_per_class, random_state=seed)
        balanced.append(sampled)

    balanced_df = pd.concat(balanced, ignore_index=True)
    balanced_df = balanced_df.sort_index()  # Restore original order

    # Verify
    print(f"\nClass distribution after sampling:")
    print(balanced_df['emotion'].value_counts().sort_index())
    print(f"\nTotal utterances: {len(balanced_df)}")

    # Verify all audio paths exist
    all_exist = balanced_df['audio_path'].apply(lambda p: Path(p).exists()).all()
    if not all_exist:
        missing = balanced_df[~balanced_df['audio_path'].apply(lambda p: Path(p).exists())]
        raise FileNotFoundError(f"Missing audio files:\n{missing[['utt_id', 'audio_path']].head()}")

    print("✓ All audio files exist")

    # Save
    balanced_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent.parent

    input_csv = root / "experiments/audio_encoder_probe/manifests/iemocap_4class.csv"
    output_csv = root / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"

    if not input_csv.exists():
        print(f"Error: {input_csv} not found")
        sys.exit(1)

    build_balanced_manifest(str(input_csv), str(output_csv))
