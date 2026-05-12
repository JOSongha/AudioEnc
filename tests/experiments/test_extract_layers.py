"""Smoke test for extract_layers.py — verifies output structure for all 4 families.

Run with: RUN_SLOW=1 pytest tests/experiments/test_extract_layers.py -v

Skipped by default (slow, requires GPU + model loading).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/audio_encoder_probe/extract_layers.py"
SMOKE_OUT = REPO / "experiments/audio_encoder_probe/embeds_layers_smoke"

LLM_LAYER_IDX = [0, 8, 15, 23, 31]
EXPECTED_KEYS = frozenset(
    [f"enc_{i}_mean" for i in range(5)]
    + [f"proj_{i}_{p}" for i in range(4) for p in ("mean", "last")]
    + [f"proj_out_{p}" for p in ("mean", "last")]
    + [f"llm_{li}_{p}" for li in LLM_LAYER_IDX for p in ("mean", "last")]
)

# Per-family expected encoder dims (matches RQ1 doc §2.3).
ENC_DIMS = {
    "whisper_tiny": [384] * 5,
    "whisper_small": [768] * 5,
    "dacvae": [64, 128, 256, 1024, 128],   # conv_in, block_0, block_1, block_3, post-VAE z
    "wavtok": [32, 64, 128, 512, 512],     # conv_in, stage_0, stage_1, stage_3, conv_out
}

PROJ_DIM = 512
PROJ_OUT_DIM = 2560
LLM_DIM = 2560


@pytest.fixture(scope="session", params=list(ENC_DIMS))
def family_outputs(request):
    """Run extract_layers for one family on 2 utterances; return list of npz paths."""
    family = request.param
    out_dir = SMOKE_OUT / family
    npz_files = list(out_dir.glob("*.npz")) if out_dir.exists() else []
    if len(npz_files) < 2:
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--family", family,
             "--out-dir", str(SMOKE_OUT),
             "--limit", "2", "--overwrite"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        if result.returncode != 0:
            pytest.fail(f"{family} extraction failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) >= 2, f"{family}: expected >=2 npz, got {len(npz_files)}"
    return family, npz_files[:2]


@pytest.mark.slow
def test_keys_match(family_outputs):
    family, files = family_outputs
    for f in files:
        keys = set(np.load(f).files)
        assert keys == EXPECTED_KEYS, (
            f"{family}/{f.name}: missing={EXPECTED_KEYS - keys}, extra={keys - EXPECTED_KEYS}")


@pytest.mark.slow
def test_enc_dims(family_outputs):
    family, files = family_outputs
    expected = ENC_DIMS[family]
    z = np.load(files[0])
    for i, exp_dim in enumerate(expected):
        actual = z[f"enc_{i}_mean"].shape
        assert actual == (exp_dim,), f"{family} enc_{i}_mean: {actual} != ({exp_dim},)"


@pytest.mark.slow
def test_proj_dims(family_outputs):
    family, files = family_outputs
    z = np.load(files[0])
    for i in range(4):
        for p in ("mean", "last"):
            assert z[f"proj_{i}_{p}"].shape == (PROJ_DIM,), \
                f"{family} proj_{i}_{p}: {z[f'proj_{i}_{p}'].shape} != ({PROJ_DIM},)"
    for p in ("mean", "last"):
        assert z[f"proj_out_{p}"].shape == (PROJ_OUT_DIM,)


@pytest.mark.slow
def test_llm_dims(family_outputs):
    family, files = family_outputs
    z = np.load(files[0])
    for li in LLM_LAYER_IDX:
        for p in ("mean", "last"):
            assert z[f"llm_{li}_{p}"].shape == (LLM_DIM,), \
                f"{family} llm_{li}_{p}: {z[f'llm_{li}_{p}'].shape} != ({LLM_DIM},)"


@pytest.mark.slow
def test_no_nan_inf(family_outputs):
    family, files = family_outputs
    for f in files:
        z = np.load(f)
        for k in z.files:
            assert np.isfinite(z[k]).all(), f"{family}/{f.name} key {k} has NaN/Inf"


@pytest.mark.slow
def test_dacvae_reproducible_z():
    """Same utt_id should produce identical post-VAE z under stable_hash seed (§12.4)."""
    out_a = SMOKE_OUT / "dacvae_seed_a"
    out_b = SMOKE_OUT / "dacvae_seed_b"
    for out in (out_a, out_b):
        if out.exists():
            for f in out.glob("*.npz"):
                f.unlink()
    for out in (out_a, out_b):
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--family", "dacvae",
             "--out-dir", str(out.parent), "--limit", "1", "--overwrite"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        # Rename out subfolder to match seed_a/seed_b
        produced = out.parent / "dacvae"
        if produced.exists() and not out.exists():
            produced.rename(out)
    files_a = sorted((out_a).glob("*.npz"))
    files_b = sorted((out_b).glob("*.npz"))
    assert files_a and files_b
    za = np.load(files_a[0])
    zb = np.load(files_b[0])
    assert np.allclose(za["enc_4_mean"], zb["enc_4_mean"], atol=1e-5), \
        "DACVAE post-VAE z not reproducible across runs (seed hash broken)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
