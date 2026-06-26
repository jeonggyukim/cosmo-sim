"""ICField: HDF5 → CIC grids → power spectra and correlations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .binning import log_bin_edges, radial_mean
from .io import read_ic_hdf5
from .windows import cic_window_squared


# ---------------------------------------------------------------------------
# CIC mass / momentum assignment (NumPy implementation)
# ---------------------------------------------------------------------------

def cic_deposit(pos: np.ndarray, boxsize: float, ngrid: int,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Single-pass CIC deposit of a scalar weight (mass by default)."""
    cellsize = boxsize / ngrid
    g = pos / cellsize
    i0 = np.floor(g).astype(int)
    dx = g - i0
    if weights is None:
        weights = np.ones(len(pos))
    out = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    for dxi in range(2):
        wx = (1 - dx[:, 0]) if dxi == 0 else dx[:, 0]
        ix = (i0[:, 0] + dxi) % ngrid
        for dyi in range(2):
            wy = (1 - dx[:, 1]) if dyi == 0 else dx[:, 1]
            iy = (i0[:, 1] + dyi) % ngrid
            for dzi in range(2):
                wz = (1 - dx[:, 2]) if dzi == 0 else dx[:, 2]
                iz = (i0[:, 2] + dzi) % ngrid
                w = wx * wy * wz * weights
                np.add.at(out, (ix, iy, iz), w)
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PowerSpectrumResult:
    """Output of ICField.power(...)."""
    k: np.ndarray            # bin centres, h/Mpc
    P: np.ndarray            # mean per-bin power (deconvolved), with shot subtracted if requested
    P_raw: np.ndarray        # mean per-bin power (deconvolved), pre-shot-subtraction
    P_nodeconv: np.ndarray   # mean per-bin power (no CIC deconv)
    sigma_P: np.ndarray      # std/sqrt(N) per bin
    nmodes: np.ndarray       # number of modes per bin
    P_shot: float            # shot noise estimate (0 if N/A)
    knyq_mpch: float         # Nyquist wavenumber, h/Mpc
    units_P: str = "(Mpc/h)^3"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ICField:
    """Loaded SWIFT IC HDF5, with cached CIC mass/momentum/velocity grids
    and Fourier-space power-spectrum / cross-spectrum machinery.

    Parameters
    ----------
    hdf5_path : str
        Path to the SWIFT IC HDF5 file.
    ngrid : int or None
        FFT grid size. If None, defaults to N^(1/3) (one cell per particle).
    interlace : bool
        If True (default), use interlaced CIC for alias suppression
        (Sefusatti et al. 2016): two grids offset by half a cell, averaged
        in Fourier space.
    h : float
        Dimensionless Hubble parameter, H0/100. Used to convert between
        Mpc and Mpc/h units in outputs.
    load_velocities : bool
        If True (default if ``ngrid`` allows), also load PartType1/Velocities
        so that velocity/momentum grids can be computed on demand.

    Attributes
    ----------
    coords : (N, 3) ndarray   particle positions in Mpc
    velocities : (N, 3) ndarray or None   peculiar velocity in km/s
    boxsize_mpc : float
    boxsize_mpch : float
    redshift : float
    h : float
    N : int                   particle count
    ngrid : int
    interlace : bool
    """

    # ----- constructor & basic accessors --------------------------------
    def __init__(self, hdf5_path: str, ngrid: int | None = None,
                 interlace: bool = True, h: float = 0.6711,
                 load_velocities: bool = True):
        d = read_ic_hdf5(hdf5_path, want_velocities=load_velocities)
        self.coords      = d["coords"]
        self.velocities  = d.get("velocities")
        self.boxsize_mpc = d["boxsize"]
        self.redshift    = d["redshift"]
        self.N           = d["N"]
        self.h           = h
        self.boxsize_mpch = self.boxsize_mpc * h
        npart_side = round(self.N ** (1.0 / 3.0))
        self.ngrid     = int(ngrid) if ngrid else int(npart_side)
        self.interlace = interlace

    # ----- k-space coordinate grids ------------------------------------
    @cached_property
    def _k_axis(self) -> np.ndarray:
        kf = 2 * np.pi / self.boxsize_mpc
        return np.fft.fftfreq(self.ngrid, d=1.0 / self.ngrid) * kf

    @cached_property
    def _kz_axis(self) -> np.ndarray:
        kf = 2 * np.pi / self.boxsize_mpc
        return np.fft.rfftfreq(self.ngrid, d=1.0 / self.ngrid) * kf

    @cached_property
    def _Kgrid(self) -> tuple:
        KX, KY, KZ = np.meshgrid(self._k_axis, self._k_axis, self._kz_axis,
                                 indexing="ij")
        return KX, KY, KZ

    @cached_property
    def _Kmag(self) -> np.ndarray:
        KX, KY, KZ = self._Kgrid
        return np.sqrt(KX**2 + KY**2 + KZ**2)

    @cached_property
    def _W2(self) -> np.ndarray:
        KX, KY, KZ = self._Kgrid
        knyq = np.pi * self.ngrid / self.boxsize_mpc
        W2 = cic_window_squared(KX, KY, KZ, knyq)
        W2[0, 0, 0] = 1.0
        return W2

    @property
    def knyq_mpc(self) -> float:
        return np.pi * self.ngrid / self.boxsize_mpc

    @property
    def knyq_mpch(self) -> float:
        return self.knyq_mpc / self.h

    @property
    def kfund_mpch(self) -> float:
        return 2 * np.pi / self.boxsize_mpch

    # ----- gridded fields -----------------------------------------------
    def _interlaced_fft(self, weights_real: np.ndarray | None = None) -> np.ndarray:
        """Interlaced-CIC FFT of mass weights ``weights_real`` (or unit per
        particle if None). Returns the real-FFT array delta_k of shape
        (ngrid, ngrid, ngrid//2+1)."""
        rho1 = cic_deposit(self.coords, self.boxsize_mpc, self.ngrid,
                                  weights_real)
        if not self.interlace:
            return np.fft.rfftn(rho1)
        dx = self.boxsize_mpc / self.ngrid
        coords2 = (self.coords + dx / 2) % self.boxsize_mpc
        rho2 = cic_deposit(coords2, self.boxsize_mpc, self.ngrid,
                                  weights_real)
        f1 = np.fft.rfftn(rho1)
        f2 = np.fft.rfftn(rho2)
        KX, KY, KZ = self._Kgrid
        phase = np.exp(1j * (KX + KY + KZ) * (dx / 2))
        return 0.5 * (f1 + phase * f2)

    @cached_property
    def density_field(self) -> np.ndarray:
        """Real-space CIC overdensity delta(x) = rho(x)/rho_bar - 1.
        (Single grid; no interlacing.)"""
        rho = cic_deposit(self.coords, self.boxsize_mpc, self.ngrid)
        return rho / (self.N / self.ngrid**3) - 1.0

    @cached_property
    def density_k(self) -> np.ndarray:
        """Fourier-space delta_k (real FFT, with interlacing if enabled)."""
        f = self._interlaced_fft()
        # Convert from "sum of weights" to overdensity in Fourier space.
        # rho_bar (in cell units) = N/Ng^3, so 1+delta = rho/(N/Ng^3).
        # In Fourier: delta_k = f_k/(N/Ng^3) - δ_{k,0} (DC mode subtraction).
        # We rely on subsequent normalisation in measure_pk; here we return
        # the FFT of (rho - rho_bar): subtract the DC value.
        f[0, 0, 0] -= self.N
        return f / (self.N / self.ngrid**3)

    @cached_property
    def momentum_k(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fourier-space CIC-deposited momentum per axis: (px_k, py_k, pz_k).
        Each is the FFT of CIC(rho * v_a), with interlacing if enabled."""
        if self.velocities is None:
            raise RuntimeError("Velocities not loaded; pass load_velocities=True.")
        outs = []
        for a in range(3):
            outs.append(self._interlaced_fft(self.velocities[:, a]))
        return tuple(outs)

    @cached_property
    def velocity_k(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fourier-space velocity per axis. Constructed by FFT of the
        real-space CIC velocity v(x) = momentum(x)/rho(x); empty cells get
        v = 0. This matches the convention used by ``compute_xi_cic``.

        Returns (vx_k, vy_k, vz_k) as rfftn-shaped arrays.
        """
        if self.velocities is None:
            raise RuntimeError("Velocities not loaded; pass load_velocities=True.")
        # Build real-space velocity grid via momentum / density (in cell-count units)
        rho_real = cic_deposit(self.coords, self.boxsize_mpc, self.ngrid)
        v_grids = []
        for a in range(3):
            mom_real = cic_deposit(self.coords, self.boxsize_mpc, self.ngrid,
                                          self.velocities[:, a])
            v = np.zeros_like(mom_real)
            mask = rho_real > 0
            v[mask] = mom_real[mask] / rho_real[mask]
            v_grids.append(v)
        # If interlacing is on, repeat with shifted grid and combine in Fourier space.
        if self.interlace:
            dx = self.boxsize_mpc / self.ngrid
            coords2 = (self.coords + dx / 2) % self.boxsize_mpc
            rho2 = cic_deposit(coords2, self.boxsize_mpc, self.ngrid)
            v_grids2 = []
            for a in range(3):
                mom2 = cic_deposit(coords2, self.boxsize_mpc, self.ngrid,
                                          self.velocities[:, a])
                v = np.zeros_like(mom2)
                mask = rho2 > 0
                v[mask] = mom2[mask] / rho2[mask]
                v_grids2.append(v)
            f1 = [np.fft.rfftn(v) for v in v_grids]
            f2 = [np.fft.rfftn(v) for v in v_grids2]
            KX, KY, KZ = self._Kgrid
            phase = np.exp(1j * (KX + KY + KZ) * (dx / 2))
            return tuple(0.5 * (a + phase * b) for a, b in zip(f1, f2))
        else:
            return tuple(np.fft.rfftn(v) for v in v_grids)

    # ----- main API: power spectrum ------------------------------------
    def power(self, field: str = "delta", *,
              nkbins: int = 60,
              kmin_mpch: float | None = None, kmax_mpch: float | None = None,
              shot_noise: bool = True) -> PowerSpectrumResult:
        """Auto power spectrum of a chosen field.

        Parameters
        ----------
        field : {"delta", "velocity"}
            Which field's power spectrum to compute.
              - "delta": density field. Returns (Mpc/h)^3 units.
              - "velocity": vector velocity power spectrum, summed over the
                three components |vx|^2 + |vy|^2 + |vz|^2. Returns
                (km/s)^2 (Mpc/h)^3 units.
        nkbins : int
            Number of log-spaced k-bins.
        kmin_mpch, kmax_mpch : float or None
            k range in h/Mpc. Defaults: fundamental mode and 0.9×Nyquist.
        shot_noise : bool
            Subtract Poisson shot noise (only meaningful for "delta").

        Returns
        -------
        PowerSpectrumResult
        """
        # Pick the per-mode |F|^2 quantity and units.
        if field == "delta":
            f_k = self.density_k
            ng3 = self.ngrid ** 3
            V_box = self.boxsize_mpc ** 3
            P_per_mode = np.abs(f_k) ** 2 * V_box / ng3**2 * self.h ** 3
            P_shot = (V_box / self.N) * self.h ** 3 if shot_noise else 0.0
            units = "(Mpc/h)^3"
        elif field == "velocity":
            vk = self.velocity_k
            ng3 = self.ngrid ** 3
            V_box = self.boxsize_mpc ** 3
            P_per_mode = sum(np.abs(v) ** 2 for v in vk) * V_box / ng3**2 * self.h ** 3
            P_shot = 0.0
            units = "(km/s)^2 (Mpc/h)^3"
        else:
            raise ValueError(f"Unknown field {field!r}; want 'delta' or 'velocity'.")

        P_nodeconv = P_per_mode.copy()
        # CIC deconvolution.
        P_per_mode = P_per_mode / self._W2

        # k in h/Mpc.
        K_mpch = self._Kmag / self.h
        kmin = kmin_mpch if kmin_mpch is not None else self.kfund_mpch
        kmax = kmax_mpch if kmax_mpch is not None else self.knyq_mpch * 0.9
        edges = log_bin_edges(kmin, kmax, nkbins)

        k_cen, P_raw, sigma_P, nmodes = radial_mean(P_per_mode, K_mpch, edges)
        _,     P_nd_mean, _,       _  = radial_mean(P_nodeconv, K_mpch, edges)

        P_ss = P_raw - P_shot if shot_noise else P_raw

        return PowerSpectrumResult(
            k=k_cen, P=P_ss, P_raw=P_raw, P_nodeconv=P_nd_mean,
            sigma_P=sigma_P, nmodes=nmodes, P_shot=P_shot,
            knyq_mpch=self.knyq_mpch, units_P=units,
        )

    # ----- summary -------------------------------------------------------
    def __repr__(self) -> str:
        v = "yes" if self.velocities is not None else "no"
        return (f"<ICField N={self.N} ({round(self.N**(1/3))}^3) "
                f"L={self.boxsize_mpc:.4g} Mpc z={self.redshift:.3g} "
                f"ngrid={self.ngrid} interlace={self.interlace} velocities={v}>")
