"""Shared HTML/audio utilities for probe analysis viewers."""


def audio_relpath(audio_path: str) -> str:
    """Convert absolute audio_path to a relative URL for HTTP serving.

    The HTML viewers under _results/predictions/ rely on these symlinks
    (created manually): nsynth_audio_root -> /mnt/tmp/datasets/music,
    audio_root -> /mnt/tmp/datasets, iemocap_audio_root -> /mnt/ddn/kyudan/IEMOCAP.

    Args:
        audio_path: Absolute filesystem path to the audio file.

    Returns:
        Relative URL (no leading slash) when path matches a known prefix;
        otherwise a ``file://`` URL as a last-resort fallback.
    """
    # nsynth + future music dataset roots live under /mnt/tmp/datasets/music/
    # (must be checked before the general /mnt/tmp/datasets/ prefix)
    if audio_path.startswith("/mnt/tmp/datasets/music/"):
        return "nsynth_audio_root/" + audio_path[len("/mnt/tmp/datasets/music/"):]
    if audio_path.startswith("/mnt/tmp/datasets/"):
        return "audio_root/" + audio_path[len("/mnt/tmp/datasets/"):]
    if audio_path.startswith("/mnt/ddn/kyudan/IEMOCAP/"):
        return "iemocap_audio_root/" + audio_path[len("/mnt/ddn/kyudan/IEMOCAP/"):]
    # fallback: file:// URL (browser may block cross-origin file access)
    return "file://" + audio_path
