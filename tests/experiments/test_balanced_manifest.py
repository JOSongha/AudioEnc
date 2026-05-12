"""Test balanced manifest generation."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def root():
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def manifest_path(root):
    return root / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"


def test_manifest_len(manifest_path):
    """Test that manifest has exactly 2000 utterances."""
    df = pd.read_csv(manifest_path)
    assert len(df) == 2000


def test_class_distribution(manifest_path):
    """Test that each emotion class has exactly 500 utterances."""
    df = pd.read_csv(manifest_path)
    counts = df['emotion'].value_counts()
    for emotion in ['angry', 'happy', 'sad', 'neutral']:
        assert counts[emotion] == 500, f"{emotion} has {counts[emotion]} utterances, expected 500"


def test_min_duration(manifest_path):
    """Test that all utterances have duration >= 1.0s."""
    df = pd.read_csv(manifest_path)
    assert df['duration_s'].min() >= 1.0


def test_all_files_exist(manifest_path):
    """Test that all audio files referenced in manifest exist."""
    df = pd.read_csv(manifest_path)
    missing = df[~df['audio_path'].apply(lambda p: Path(p).exists())]
    assert len(missing) == 0, f"Missing files: {missing['utt_id'].tolist()}"


def test_deterministic_sampling(root):
    """Test that sampling is deterministic (seed=42)."""
    from experiments.audio_encoder_probe.build_balanced_manifest import build_balanced_manifest
    import tempfile

    input_csv = root / "experiments/audio_encoder_probe/manifests/iemocap_4class.csv"

    # Generate twice with same seed
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f1:
        out1 = f1.name
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f2:
        out2 = f2.name

    try:
        build_balanced_manifest(str(input_csv), out1, n_per_class=500, seed=42)
        build_balanced_manifest(str(input_csv), out2, n_per_class=500, seed=42)

        df1 = pd.read_csv(out1)
        df2 = pd.read_csv(out2)

        # Same utterances
        assert set(df1['utt_id']) == set(df2['utt_id']), "Sampling not deterministic"
    finally:
        Path(out1).unlink(missing_ok=True)
        Path(out2).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
