"""Mass-assignment window functions and deconvolution helpers."""

from __future__ import annotations

import numpy as np

# Interpolation order p for each mass-assignment scheme (Sefusatti et al. 2016):
# NGP=1, CIC=2, TSC=3, PCS=4. The per-axis Fourier window is sinc(k/2k_Ny)^p,
# so the squared window used to deconvolve |delta(k)|^2 is sinc(...)^(2p).
ASSIGNMENT_ORDER = {"ngp": 1, "cic": 2, "tsc": 3, "pcs": 4}


def assignment_window_squared(KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray,
                              knyq: float, order: int = 2) -> np.ndarray:
    """Squared mass-assignment window, prod_a [sinc(k_a / 2 k_Ny)]^(2*order).

    ``order`` is the interpolation order p (NGP=1, CIC=2, TSC=3, PCS=4). Dividing
    |delta(k)|^2 by this corrects for the assignment smoothing. Higher order also
    shrinks the residual aliasing images (Sefusatti et al. 2016), especially with
    interlacing on.
    """
    def s(k):
        return np.sinc(k / (2.0 * knyq))
    return (s(KX) * s(KY) * s(KZ)) ** (2 * order)


def cic_window_squared(KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray,
                       knyq: float) -> np.ndarray:
    """Squared CIC window function.

    For Cloud-in-Cell (CIC, NGP-of-order-2) mass assignment on a grid
    with Nyquist wavenumber ``knyq``, the smoothing kernel is a top-hat
    of one cell width, whose Fourier-space response per axis is
    ``sinc(k_a / 2 k_Ny)``. CIC is degree-2 (linear interpolation), so
    the per-axis power is the square of that, and the 3D squared window
    is the product over axes:

        W^2(k) = prod_{a=x,y,z} [sinc(k_a / 2 k_Ny)]^4 .

    Dividing |delta(k)|^2 by W^2 corrects for the CIC smoothing.

    Parameters
    ----------
    KX, KY, KZ : ndarray
        k-component grids (any shape, broadcasted together) in the same
        units as ``knyq``.
    knyq : float
        Nyquist wavenumber, ``pi * Ngrid / L`` in those units.

    Returns
    -------
    W2 : ndarray
        Squared CIC window, same shape as the broadcast of (KX, KY, KZ).
        Strictly positive on the resolved-mode region; tiny near the
        corners of the Brillouin zone.

    Notes
    -----
    Uses ``numpy.sinc(u) = sin(pi u) / (pi u)`` (the normalised sinc),
    which is well-defined at the origin (returns 1).
    """
    def s(k):
        return np.sinc(k / (2.0 * knyq))
    return (s(KX) * s(KY) * s(KZ)) ** 4
