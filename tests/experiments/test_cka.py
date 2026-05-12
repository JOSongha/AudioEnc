"""Unit tests for linear CKA implementation in compute_cka.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/audio_encoder_probe"))

from compute_cka import center, linear_cka  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_self_similarity(rng):
    """CKA(X, X) = 1.0."""
    X = rng.standard_normal((200, 64)).astype(np.float32)
    assert linear_cka(center(X), center(X)) == pytest.approx(1.0, abs=1e-5)


def test_symmetry(rng):
    """CKA(X, Y) == CKA(Y, X)."""
    X = rng.standard_normal((200, 32)).astype(np.float32)
    Y = rng.standard_normal((200, 48)).astype(np.float32)
    Xc, Yc = center(X), center(Y)
    assert linear_cka(Xc, Yc) == pytest.approx(linear_cka(Yc, Xc), abs=1e-6)


def test_random_baseline(rng):
    """Two independent Gaussian sets should have CKA near 0 (well below 0.1)."""
    X = rng.standard_normal((2000, 128)).astype(np.float32)
    Y = rng.standard_normal((2000, 128)).astype(np.float32)
    val = linear_cka(center(X), center(Y))
    assert 0.0 <= val < 0.1, f"random baseline {val} not in [0, 0.1)"


def test_different_dims(rng):
    """CKA accepts (N, D1) and (N, D2) with D1 != D2."""
    X = rng.standard_normal((300, 32)).astype(np.float32)
    Y = rng.standard_normal((300, 512)).astype(np.float32)
    val = linear_cka(center(X), center(Y))
    assert 0.0 <= val <= 1.0


def test_translation_invariance(rng):
    """Centering removes translation: CKA(X, Y) == CKA(X + c, Y + d)."""
    X = rng.standard_normal((200, 64)).astype(np.float32)
    Y = rng.standard_normal((200, 64)).astype(np.float32)
    val_a = linear_cka(center(X), center(Y))
    val_b = linear_cka(center(X + 5.0), center(Y - 3.0))
    assert val_a == pytest.approx(val_b, abs=1e-6)


def test_linear_transform_invariance(rng):
    """CKA is invariant to orthogonal transforms on either side (Kornblith 2019, Thm 1)."""
    X = rng.standard_normal((200, 64)).astype(np.float32)
    Y = rng.standard_normal((200, 64)).astype(np.float32)
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)).astype(np.float32))
    val_a = linear_cka(center(X), center(Y))
    val_b = linear_cka(center(X @ Q), center(Y))
    assert val_a == pytest.approx(val_b, abs=1e-5)


def test_range_bounded(rng):
    """CKA values must be in [0, 1]."""
    for _ in range(5):
        X = rng.standard_normal((150, 64)).astype(np.float32)
        Y = rng.standard_normal((150, 32)).astype(np.float32)
        val = linear_cka(center(X), center(Y))
        assert 0.0 <= val <= 1.0


def test_correlated_features(rng):
    """If Y = X + small noise, CKA should be near 1."""
    X = rng.standard_normal((500, 32)).astype(np.float32)
    Y = X + 0.01 * rng.standard_normal((500, 32)).astype(np.float32)
    val = linear_cka(center(X), center(Y))
    assert val > 0.99, f"correlated CKA {val} not > 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
