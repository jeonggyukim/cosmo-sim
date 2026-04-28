"""Tests for icpipe.theory.LinearTheory."""

import numpy as np
import pytest

from icpipe.theory import LinearTheory


@pytest.fixture
def th():
    """Synthetic toy LinearTheory: P(k) = k^-2 over a wide k range."""
    k = np.logspace(-3, 1, 400)
    return LinearTheory(k_table=k, Pk_table=k**(-2.0), z=0.0,
                        h=0.7, Omega_m=0.3)


def test_growth_rate_z0(th):
    # f ≈ Omega_m^0.55 ≈ 0.3^0.55 ≈ 0.512
    assert np.isclose(th.f_growth, 0.3**0.55, rtol=1e-6)


def test_pk_interpolation(th):
    # On the table itself, Pk(k_table) recovers k_table^-2 exactly.
    k = np.array([0.01, 0.1, 1.0])
    np.testing.assert_allclose(th.Pk(k), k**(-2.0), rtol=1e-3)


def test_pv_kernel_relation(th):
    """P_v(k)/P_delta(k) should equal (a H f / k)^2 in (km/s)^2 (Mpc/h)^2."""
    k = np.array([0.05, 0.5])
    Pv = th.Pv(k)
    Pk = th.Pk(k)
    expected = (th.a * th.H_kmsMpch * th.f_growth / k) ** 2
    np.testing.assert_allclose(Pv / Pk, expected, rtol=1e-10)


def test_psi_at_zero_separation_consistent_with_variance(th):
    # psi(r=0) = sigma_v^2; finite for our P(k) ~ k^-2 (variance integral
    # is int k^0 P(k) dk = int dk, divergent without a cutoff). Use a
    # narrower table to keep the integral finite.
    k = np.logspace(-2, 1, 400)
    th2 = LinearTheory(k_table=k, Pk_table=np.exp(-k**2), z=0.0,
                       h=0.7, Omega_m=0.3)
    psi0 = th2.psi(np.array([1e-3]))[0]  # very small r ≈ 0
    expected = (th2.aHf / th2.h) ** 2 / (2 * np.pi**2) * np.trapezoid(np.exp(-k**2), k)
    np.testing.assert_allclose(psi0, expected, rtol=2e-3)
