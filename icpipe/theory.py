"""Linear-theory predictions for P_delta, P_v, xi, psi.

The single ``LinearTheory`` class loads a CLASS-output P(k) table and
provides interpolated linear-theory observables in consistent units:

    P_δ(k)   in (Mpc/h)^3
    P_v(k)   in (km/s)^2 (Mpc/h)^3
    ξ(r)     dimensionless,  Hankel transform of P_δ
    ψ(r)     in (km/s)^2,    Hankel transform of P_v

All four are derived from the same ``Pk_table`` and the same growth-rate
expression so any overlay against measurement is internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearTheory:
    """Linear-theory predictions from a CLASS P(k) table.

    Parameters
    ----------
    k_table : ndarray
        Wavenumbers in h/Mpc (typically log-spaced from CLASS).
    Pk_table : ndarray
        Linear matter power spectrum in (Mpc/h)^3 at the same redshift.
    z : float
        Redshift the P(k) was evaluated at.
    h : float
        Dimensionless Hubble parameter, H0/100.
    Omega_m : float
        Matter density parameter today.
    """
    k_table: np.ndarray
    Pk_table: np.ndarray
    z: float
    h: float = 0.6711
    Omega_m: float = 0.3158

    # ----- constructors ------------------------------------------------
    @classmethod
    def from_class(cls, path: str, *, z: float, h: float = 0.6711,
                   Omega_m: float = 0.3158) -> "LinearTheory":
        """Load a CLASS ``class_pk_z*_pk.dat`` two-column file."""
        kt, Pt = np.loadtxt(path, comments="#", unpack=True)
        return cls(k_table=kt, Pk_table=Pt, z=z, h=h, Omega_m=Omega_m)

    @classmethod
    def from_class_backscaled(cls, z0_path: str, *, z_target: float,
                              h: float = 0.6711,
                              Omega_m: float = 0.3158) -> "LinearTheory":
        """Load a CLASS table assumed at z=0 and back-scale to z_target via

            P(k, z_target) = D_norad(z_target)^2 * P_CLASS(k, z=0)

        using the no-radiation linear growth factor (matches MUSIC2's
        ``ZeroRadiation = true`` convention; see ``D_norad``). Returns a
        LinearTheory whose ``.z = z_target`` and ``.Pk(k)`` already gives
        the back-scaled prediction.
        """
        kt, Pt = np.loadtxt(z0_path, comments="#", unpack=True)
        tmp = cls(k_table=kt, Pk_table=Pt, z=0.0, h=h, Omega_m=Omega_m)
        D2 = tmp.D_norad(z_target) ** 2
        return cls(k_table=kt, Pk_table=Pt * D2, z=z_target,
                   h=h, Omega_m=Omega_m)

    # ----- background scalars -----------------------------------------
    @property
    def a(self) -> float:
        return 1.0 / (1.0 + self.z)

    @property
    def Ez(self) -> float:
        """E(z) = H(z)/H0 in flat LCDM (matter + cosmological constant)."""
        return float(np.sqrt(self.Omega_m * (1 + self.z)**3
                             + (1 - self.Omega_m)))

    @property
    def H0_kmsMpc(self) -> float:
        return 100.0 * self.h

    @property
    def H_kmsMpc(self) -> float:
        """H(z) in km/s/Mpc."""
        return self.H0_kmsMpc * self.Ez

    @property
    def H_kmsMpch(self) -> float:
        """H(z) in km/s/(Mpc/h) — convenient for k in h/Mpc."""
        return self.H_kmsMpc / self.h

    @property
    def f_growth(self) -> float:
        """Linear growth rate f ≈ Omega_m(z)^0.55 (Linder 2005)."""
        return float((self.Omega_m * (1 + self.z)**3 / self.Ez**2) ** 0.55)

    def D_norad(self, z: float | None = None) -> float:
        """Linear growth factor D+(z) in flat ΛCDM with Omega_r=0,
        normalised to D+(0)=1. Matches MUSIC2's ``ZeroRadiation=true``
        convention (and the assumption made by SWIFT/GADGET in their
        background Friedmann evolution).

            D+(z) ∝ H(a) * ∫_0^a da' / [a' H(a')]^3
        """
        from scipy.integrate import quad
        if z is None:
            z = self.z
        Om, OL = self.Omega_m, 1.0 - self.Omega_m

        def H_over_H0(a):
            return np.sqrt(Om / a**3 + OL)

        def integrand(a):
            return 1.0 / (a * H_over_H0(a)) ** 3

        a = 1.0 / (1.0 + z)
        D_a, _ = quad(integrand, 1e-6, a,   limit=500)
        D_0, _ = quad(integrand, 1e-6, 1.0, limit=500)
        return float((H_over_H0(a) * D_a) / (H_over_H0(1.0) * D_0))

    @property
    def aHf(self) -> float:
        """a * H(z) * f(z) in km/s/Mpc."""
        return self.a * self.H_kmsMpc * self.f_growth

    # ----- spectra -----------------------------------------------------
    def Pk(self, k: np.ndarray) -> np.ndarray:
        """Linear matter P_delta(k) interpolated to query wavenumbers ``k``
        (h/Mpc). Outside the table range, returns 0.
        """
        k = np.asarray(k, dtype=float)
        out = np.zeros_like(k)
        m = (k >= self.k_table.min()) & (k <= self.k_table.max())
        out[m] = np.exp(np.interp(np.log(k[m]),
                                  np.log(self.k_table),
                                  np.log(self.Pk_table)))
        return out

    def Pv(self, k: np.ndarray) -> np.ndarray:
        """Linear velocity power spectrum

            P_v(k) = (a H f / k)^2 * P_delta(k)

        with k in h/Mpc and result in (km/s)^2 (Mpc/h)^3. The (a H f) is
        in km/s/(Mpc/h), so the dimensional factor (aHf/k)^2 has units
        (km/s)^2 / (Mpc/h)^{-2} ... wait — (km/s)^2 (Mpc/h)^2 — and
        multiplied by P_delta in (Mpc/h)^3 gives (km/s)^2 (Mpc/h)^5; divide
        by k^2 in (h/Mpc)^2 gives (km/s)^2 (Mpc/h)^3. ✓
        """
        Pdelta = self.Pk(k)
        prefac = (self.a * self.H_kmsMpch * self.f_growth) ** 2
        out = np.zeros_like(np.asarray(k, dtype=float))
        m = np.asarray(k) > 0
        kk = np.asarray(k, dtype=float)
        out[m] = prefac * Pdelta[m] / kk[m] ** 2
        return out

    def Ptheta(self, k: np.ndarray) -> np.ndarray:
        """Linear velocity-divergence power spectrum
        P_theta(k) = (a H f)^2 P_delta(k), in (km/s)^2 (Mpc/h)^3 (per unit
        k^2 of the vector spectrum)."""
        return (self.a * self.H_kmsMpch * self.f_growth) ** 2 * self.Pk(k)

    # ----- Hankel transforms -------------------------------------------
    def _hankel(self, r_vals: np.ndarray, integrand: np.ndarray,
                k_grid: np.ndarray | None = None) -> np.ndarray:
        """Generic spherical Hankel transform with j_0:
            f(r) = (1/(2 pi^2)) int dk integrand(k) j_0(k r)

        ``integrand`` must already include the "k^2 P(k)" factor for
        density-style transforms or just P(k) for velocity-style ones.
        """
        if k_grid is None:
            k_grid = self.k_table
        r_vals = np.asarray(r_vals, dtype=float)
        out = np.empty_like(r_vals)
        for i, ri in enumerate(r_vals):
            x = k_grid * ri
            j0 = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
            out[i] = np.trapezoid(integrand * j0, k_grid) / (2 * np.pi**2)
        return out

    def xi(self, r: np.ndarray) -> np.ndarray:
        """Linear two-point correlation function
            xi(r) = (1/(2 pi^2)) int k^2 P_delta(k) j_0(k r) dk
        with r in Mpc/h."""
        return self._hankel(r, self.k_table**2 * self.Pk_table)

    def psi(self, r: np.ndarray) -> np.ndarray:
        """Linear peculiar-velocity correlation function
            psi(r) = (a H f)^2 / (2 pi^2) int P_delta(k) j_0(k r) dk
        with r in Mpc/h, result in (km/s)^2.

        Note that the integrand is P(k) — no k^2 factor — because the
        velocity-density kernel is i k/k^2 (the k^2 measure of the volume
        element cancels the 1/k^2 of the kernel)."""
        prefac = (self.a * self.H_kmsMpch * self.f_growth) ** 2
        # Hankel without the k^2 factor:
        out = self._hankel(r, self.Pk_table)
        return prefac * out
