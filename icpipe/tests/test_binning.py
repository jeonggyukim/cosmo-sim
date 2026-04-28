"""Tests for icpipe.binning."""

import numpy as np

from icpipe.binning import lin_bin_edges, log_bin_edges, radial_mean


def test_log_bin_edges_count():
    e = log_bin_edges(0.01, 1.0, 10)
    assert len(e) == 11
    assert np.isclose(e[0], 0.01)
    assert np.isclose(e[-1], 1.0)


def test_lin_bin_edges_uniform():
    e = lin_bin_edges(0, 10, 5)
    np.testing.assert_allclose(np.diff(e), 2.0)


def test_radial_mean_uniform():
    """Constant-valued field should return the same constant in every bin."""
    K = np.linspace(0.1, 1.0, 100)
    V = np.full_like(K, 7.0)
    edges = log_bin_edges(0.1, 1.0, 5)
    k_cen, mean, err, n = radial_mean(V, K, edges)
    np.testing.assert_allclose(mean, 7.0)
    assert (n > 0).all()


def test_radial_mean_excludes_dc():
    """The k=0 mode must be excluded."""
    K = np.array([0.0, 0.5, 0.6, 0.7])
    V = np.array([1e10, 1.0, 1.0, 1.0])  # huge weight on DC
    edges = log_bin_edges(0.1, 1.0, 3)
    _, mean, _, _ = radial_mean(V, K, edges)
    assert (mean < 1e9).all()  # DC mode would push mean to ~1e10 if included


def test_radial_mean_drops_empty_bins():
    K = np.array([0.5, 0.6])
    V = np.array([1.0, 1.0])
    edges = np.array([0.0, 0.4, 0.7, 1.0])  # middle bin only
    k_cen, mean, err, n = radial_mean(V, K, edges)
    assert len(k_cen) == 1
    assert n[0] == 2
