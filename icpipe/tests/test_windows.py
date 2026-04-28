"""Tests for icpipe.windows."""

import numpy as np
import pytest

from icpipe.windows import cic_window_squared


def test_window_unity_at_origin():
    """W^2(k=0) = 1 (no smoothing of the DC mode)."""
    KX = np.array([0.0])
    KY = np.array([0.0])
    KZ = np.array([0.0])
    assert np.isclose(cic_window_squared(KX, KY, KZ, knyq=1.0), 1.0)


def test_window_axis_nyquist():
    """W^2(k_Ny along one axis, others zero) = sinc^4(1/2) = (2/pi)^4."""
    KX = np.array([1.0])  # = knyq
    KY = np.array([0.0])
    KZ = np.array([0.0])
    expected = (2.0 / np.pi) ** 4
    actual = cic_window_squared(KX, KY, KZ, knyq=1.0)[0]
    assert np.isclose(actual, expected)


def test_window_separability():
    """W^2(kx,ky,kz) = W^2_x * W^2_y * W^2_z."""
    rng = np.random.default_rng(42)
    K = rng.uniform(-2, 2, size=(20,))
    knyq = 1.0
    K0 = np.zeros_like(K)
    Wx = cic_window_squared(K, K0, K0, knyq=knyq)
    Wy = cic_window_squared(K0, K, K0, knyq=knyq)
    Wz = cic_window_squared(K0, K0, K, knyq=knyq)
    # Each per-axis W^2 evaluated this way is sinc^4(k/2) (since the other
    # axes contribute 1). The 3D product W^2(k,k,k) for the same k along
    # all three axes should be Wx * Wy * Wz / 1 / 1 ... not quite — let's
    # test the simpler factorisation property directly.
    Wxyz = cic_window_squared(K, K, K, knyq=knyq)
    expected = Wx * Wy * Wz
    np.testing.assert_allclose(Wxyz, expected, rtol=1e-12)


def test_window_positive():
    """W^2 >= 0 inside the fine-mode region."""
    rng = np.random.default_rng(0)
    KX = rng.uniform(-1, 1, size=50)
    KY = rng.uniform(-1, 1, size=50)
    KZ = rng.uniform(-1, 1, size=50)
    W2 = cic_window_squared(KX, KY, KZ, knyq=1.0)
    assert (W2 >= 0).all()
