"""Tests for audio_relpath path-rewriting."""

from experiments.audio_encoder_probe._html_utils import audio_relpath


def test_nsynth_prefix() -> None:
    out = audio_relpath("/mnt/tmp/datasets/music/nsynth/train_30k/foo.wav")
    assert out == "nsynth_audio_root/nsynth/train_30k/foo.wav"


def test_general_datasets_prefix() -> None:
    out = audio_relpath("/mnt/tmp/datasets/esc50/audio/1-100032-A-0.wav")
    assert out == "audio_root/esc50/audio/1-100032-A-0.wav"


def test_iemocap_prefix() -> None:
    out = audio_relpath("/mnt/ddn/kyudan/IEMOCAP/Session1/foo.wav")
    assert out == "iemocap_audio_root/Session1/foo.wav"


def test_unknown_prefix_falls_back_to_file_url() -> None:
    out = audio_relpath("/some/other/path/audio.wav")
    assert out == "file:///some/other/path/audio.wav"


def test_nsynth_takes_precedence_over_general() -> None:
    # /mnt/tmp/datasets/music/ 가 /mnt/tmp/datasets/ 보다 먼저 매치되어야 함
    out = audio_relpath("/mnt/tmp/datasets/music/x.wav")
    assert out.startswith("nsynth_audio_root/")
