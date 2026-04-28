"""Radial binning utilities used by the ICField power-spectrum routines."""

from __future__ import annotations

import numpy as np


def log_bin_edges(kmin: float, kmax: float, nbins: int) -> np.ndarray:
    """Logarithmically spaced bin edges from ``kmin`` to ``kmax`` (nbins+1 edges)."""
    return np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)


def lin_bin_edges(kmin: float, kmax: float, nbins: int) -> np.ndarray:
    """Linearly spaced bin edges from ``kmin`` to ``kmax`` (nbins+1 edges)."""
    return np.linspace(kmin, kmax, nbins + 1)


def radial_mean(values: np.ndarray, K: np.ndarray, edges: np.ndarray):
    """Bin ``values`` by ``|K|`` using the given bin edges.

    Parameters
    ----------
    values : ndarray
        Per-mode values to be averaged inside each bin (e.g., |delta(k)|^2 / W^2).
    K : ndarray
        Per-mode wavenumber magnitudes, same shape as ``values``.
    edges : ndarray, shape (nbins+1,)
        Monotonically increasing bin edges.

    Returns
    -------
    k_cen : ndarray, shape (M,)
        Mean of ``|K|`` over the modes that fell into each non-empty bin.
    mean : ndarray, shape (M,)
        Mean of ``values`` over the modes in each non-empty bin.
    err : ndarray, shape (M,)
        Standard error of the mean (std/sqrt(N)) per bin.
    nmodes : ndarray, shape (M,)
        Number of modes per bin.

    Notes
    -----
    Empty bins are dropped; the returned arrays have length M ≤ nbins.
    The k=0 mode is excluded.
    """
    Kf = K.ravel()
    Vf = values.ravel()
    mask = Kf > 0
    Kf = Kf[mask]
    Vf = Vf[mask]
    nbins = len(edges) - 1
    k_cen  = np.zeros(nbins)
    mean   = np.zeros(nbins)
    err    = np.zeros(nbins)
    nmodes = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        sel = (Kf >= edges[i]) & (Kf < edges[i+1])
        n = int(sel.sum())
        if n > 0:
            k_cen[i]  = Kf[sel].mean()
            mean[i]   = Vf[sel].mean()
            err[i]    = Vf[sel].std() / np.sqrt(n) if n > 1 else 0.0
            nmodes[i] = n
    keep = nmodes > 0
    return k_cen[keep], mean[keep], err[keep], nmodes[keep]
