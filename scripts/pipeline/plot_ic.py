#!/usr/bin/env python3
"""
plot_ic.py — Plot P(k) and ξ(r)/ψ(r) from saved pk_*.txt files.

Reads the output of compute_pk.py (pk_*.txt) and, if present in the same
directory, overlays:
  - Corrfunc xi(r) from xi_<stem>.txt
  - CIC grid xi(r) from xi_cic_<stem>.txt
  - CIC velocity correlation ψ(r) from vel_cic_<stem>.txt
  - CLASS theory P(k), ξ(r), ψ(r)

Usage:
    # Single run (auto-detects xi/vel files alongside pk_*.txt)
    python plot_ic.py data/pk_n256_z2_L687.txt --theory data/class_pk_z2_pk.dat

    # Multiple runs overlaid
    python plot_ic.py data/pk_n256_z45_L500.txt data/pk_n512_z45_L500.txt \\
        --theory data/class_pk_z45_pk.dat

    # Explicit supplementary files
    python plot_ic.py data/pk_n256_z2_L687.txt \\
        --xi-cic data/xi_cic_n256_z2_L687.txt \\
        --vel-cic data/vel_cic_n256_z2_L687.txt \\
        --theory data/class_pk_z2_pk.dat -o plots/pk_z2.png

Output units: k in h/Mpc, P(k) in (Mpc/h)³, r in Mpc/h, ψ(r) in (km/s)².
"""

import argparse
import os
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# PKPlotter class
# ---------------------------------------------------------------------------

class ICPlotter:
    """
    Load and plot P(k) / ξ(r) / ψ(r) diagnostics for SWIFT IC runs.

    Typical usage::

        plotter = ICPlotter(H0=67.11, Omega_m=0.3, Omega_b=0.049)
        plotter.load_pk_file("data/pk_n256_z2_L687.txt")
        plotter.load_theory("data/class_pk_z2_pk.dat")
        plotter.auto_load_xi()       # finds xi/vel_cic files automatically
        plotter.plot()
        plotter.save("plots/pk_n256_z2_L687.png")
    """

    def __init__(self, H0=67.11, Omega_m=0.3, Omega_b=0.049):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b

        # Loaded data
        self.pk_runs = []     # list of dicts, one per pk_*.txt
        self.theory_k = None   # CLASS k [h/Mpc]
        self.theory_P = None   # CLASS P(k) [(Mpc/h)³]
        self.theory_xi_r = None   # theory ξ(r) from Hankel transform
        self.theory_xi = None
        self.theory_psi_r = None   # theory ψ(r) [(km/s)²]
        self.theory_psi = None
        self.corrfunc_xi = None   # dict: r_mid, xi, err, nseeds_used
        self.xi_cic_list = []     # list of dicts: r_mid, xi, stem (one per pk run)
        self.vel_cic_list = []     # list of dicts: r_mid, psi, stem (one per pk run)

        # Figure handles (set by plot())
        self.fig = self.ax = self.ax2 = self.ax3 = None
        self.ax_pk_ratio = self.ax_xi_ratio = None

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_header(path):
        """Extract BoxSize [Mpc/h], N, P_shot from pk_*.txt header."""
        meta = {"boxsize_mpch": None, "N": None, "P_shot": None}
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                m = re.search(r"BoxSize=([\d.e+\-]+)\s*Mpc/h", line)
                if m:
                    meta["boxsize_mpch"] = float(m.group(1))
                m = re.search(r"\bN=(\d+)\b", line)
                if m:
                    meta["N"] = int(m.group(1))
                m = re.search(r"P_shot\s*=\s*([\d.e+\-]+)", line)
                if m:
                    meta["P_shot"] = float(m.group(1))
        return meta

    @staticmethod
    def _stem_from_pkfile(pkfile):
        return os.path.basename(pkfile).replace("pk_", "").replace(".txt", "")

    @staticmethod
    def _label_from_stem(stem):
        """Format stem 'n256_z200_L172' → 'N256_L172_Z200'."""
        mn = re.search(r'n(\d+)', stem)
        mz = re.search(r'_z([\d.]+)', stem)
        ml = re.search(r'_L(\d+)', stem)
        parts = []
        if mn:
            parts.append(f'N{mn.group(1)}')
        if ml:
            parts.append(f'L{ml.group(1)}')
        if mz:
            parts.append(f'Z{mz.group(1)}')
        return '_'.join(parts) if parts else stem

    @staticmethod
    def _parse_z_from_stem(stem):
        """Extract redshift from stem like 'n256_z2_L687' → 2.0."""
        m = re.search(r'_z([\d.]+)_', stem) or re.search(r'_z([\d.]+)$', stem)
        return float(m.group(1)) if m else None

    @staticmethod
    def _bao_scales(H0=67.11, Omega_m=0.3, Omega_b=0.049):
        h = H0 / 100.0
        Omh2 = Omega_m * h**2
        Obh2 = Omega_b * h**2
        r_d_Mpc = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(3/4))
        r_d_mpch = r_d_Mpc * h
        return r_d_mpch, 2 * np.pi / r_d_mpch

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_pk_file(self, pkfile):
        """Load one pk_*.txt file (output of compute_pk.py)."""
        meta = self._parse_header(pkfile)
        data = np.loadtxt(pkfile, comments='#')
        # Columns: k  P_raw  P_shot_sub  P_nodeconv  sigma_P  nmodes  [fold_m]
        run = {
            "pkfile":           pkfile,
            "stem":             self._stem_from_pkfile(pkfile),
            "data_dir":         os.path.dirname(os.path.abspath(pkfile)),
            "k":                data[:, 0],
            "Pk_raw":           data[:, 1],
            "Pk_ss":            data[:, 2],
            "Pk_nodeconv":      data[:, 3] if data.shape[1] > 3 else data[:, 1],
            "Pk_err":           data[:, 4] if data.shape[1] > 4 else np.zeros(len(data)),
            "fold_m":           (data[:, 6].astype(int) if data.shape[1] > 6
                                 else np.ones(len(data), dtype=int)),
            "P_shot":           meta["P_shot"],
            "boxsize_mpch":     meta["boxsize_mpch"],
            "N":                meta["N"],
            "npart_side":       round(meta["N"]**(1/3)) if meta["N"] else None,
            "z":                self._parse_z_from_stem(self._stem_from_pkfile(pkfile)),
        }
        self.pk_runs.append(run)

    @staticmethod
    def _growth_factor_norad(z, Omega_m, H0):
        """
        Linear growth factor D+(z) with Omega_r = 0 (ZeroRadiation convention),
        normalised to D+(z=0) = 1.  Matches MUSIC2's back-scaling growth factor.

        Uses the standard integral form:
            D+(z) ∝ H(z) ∫_0^a da' / [a' H(a')]³
        with H²(a) = H0² [Omega_m/a³ + (1 - Omega_m)].
        """
        from scipy.integrate import quad
        Omega_DE = 1.0 - Omega_m  # flat, no radiation

        def H(a):
            return H0 * np.sqrt(Omega_m / a**3 + Omega_DE)

        def integrand(a):
            return 1.0 / (a * H(a))**3
        a = 1.0 / (1.0 + z)
        D, _ = quad(integrand, 1e-6, a,   limit=500)
        D0, _ = quad(integrand, 1e-6, 1.0, limit=500)
        return (H(a) * D) / (H(1.0) * D0)

    def load_theory(self, theory_file, z_ref=None):
        """
        Load CLASS P(k) and compute theory ξ(r) and ψ(r).

        If z_ref is given, the theory file is treated as P(k) at redshift
        z_ref and back-scaled to z_start (from the primary pk run) using the
        no-radiation growth factor D+_norad — matching MUSIC2's ZeroRadiation=true
        convention.  Use this when comparing high-z ICs against CLASS:

            P_ref(k, z_start) = P_CLASS(k, z_ref) × [D+_norad(z_start) / D+_norad(z_ref)]²

        Using z_ref=0 is the natural choice: D+_norad(0)=1 by definition, so
        the formula simplifies to P_ref(k, z_start) = P_CLASS(k, 0) × D+_norad(z_start)².

        MUSIC2 uses Omega_r=0 (ZeroRadiation=true) because downstream N-body
        /hydro codes (SWIFT, GADGET, ...) ignore radiation in their background
        Friedmann evolution; the ICs must share that approximation or the
        mode amplitudes at z_start will disagree with what the integrator
        expects.  Boltzmann codes (CLASS, CAMB) evolve with radiation on,
        so their D+(z) differs by ~11% at z=200 — hence the ~5-10% P(k)
        suppression one sees without back-scaling.

        ξ(r) is computed via Hankel transform using mcfit (if available).
        ψ(r) = [H(z)·f(z)]²/(2π²) ∫ P(k) j₀(kr) dk is computed via direct
        trapezoid integration.  Requires that at least one pk run has been
        loaded (to determine z).
        """
        kt, Pt = np.loadtxt(theory_file, comments='#', unpack=True)

        if z_ref is not None and self.pk_runs:
            z_start = self.pk_runs[0].get("z")
            if z_start is not None and z_start != z_ref:
                D_start = self._growth_factor_norad(z_start, self.Omega_m, self.H0)
                D_ref = self._growth_factor_norad(z_ref,   self.Omega_m, self.H0)
                scale = (D_start / D_ref) ** 2
                Pt = Pt * scale
                print(f"Theory back-scaled from z_ref={z_ref} to z_start={z_start:.4g} "
                      f"using no-radiation D+: scale factor = {scale:.6g}")

        self.theory_k = kt
        self.theory_P = Pt

        # Theory ξ(r) via Hankel transform
        try:
            from mcfit import P2xi
            r, xi = P2xi(kt, l=0)(Pt, extrap=True)
            self.theory_xi_r = r
            self.theory_xi = xi
        except ImportError:
            pass

        # Theory ψ(r): need z from the primary pk run
        if self.pk_runs:
            z = self.pk_runs[0].get("z")
            if z is not None:
                Ez = np.sqrt(self.Omega_m * (1+z)**3 + (1 - self.Omega_m))
                Hz = self.H0 * Ez           # km/s/Mpc
                Hz_hMpc = Hz / self.h            # km/s/(Mpc/h) — consistent with k in h/Mpc
                fz = (self.Omega_m * (1+z)**3 / Ez**2) ** 0.55   # Linder 2005
                a = 1.0 / (1.0 + z)
                r_grid = self.theory_xi_r if self.theory_xi_r is not None else None
                r_psi, psi = self._compute_theory_psi(kt, Pt, a, Hz_hMpc, fz,
                                                      r_grid=r_grid)
                self.theory_psi_r = r_psi
                self.theory_psi = psi
                print(f"Theory ψ(r): z={z:.4g}, a={a:.4g}, H={Hz:.1f} km/s/Mpc, "
                      f"f={fz:.4f}, aHf/(h)={a*Hz_hMpc*fz:.2f} km/s/(Mpc/h)")

    @staticmethod
    def _compute_theory_psi(kt, Pt, a, Hz_hMpc, fz, r_grid=None):
        """
        Compute the true peculiar-velocity correlation ψ(r) = ⟨v_pec(x)·v_pec(x+r)⟩.

        In linear theory the proper peculiar velocity is irrotational with
            v_pec,k = i (a·H·f/k) k̂ δ_k
        (since v_pec = a·ẋ_comoving and δ̇ = -∇·v_pec/a = H·f·δ for ZA).
        The isotropic scalar correlation is:
            ψ_true(r) = [a·H·f]²/(2π²) ∫ P(k) j₀(kr) dk

        The k²/k² factor (from the velocity power spectrum vs the angular
        measure) cancels, unlike ξ(r) which has an extra k² in the integrand.

        MUSIC2's SWIFT plugin writes the on-disk Velocities dataset in
        peculiar-velocity units (km/s), so the CIC correlation from
        compute_xi_cic is already ψ_pec — no a-rescaling needed on the
        measurement side either.

        Parameters
        ----------
        kt, Pt  : CLASS P(k): k in h/Mpc, P in (Mpc/h)³
        a       : scale factor 1/(1+z) at the IC redshift
        Hz_hMpc : H(z) in km/s/(Mpc/h)
        fz      : growth rate f ≈ Ω_m(z)^0.55

        Returns
        -------
        r   : array [Mpc/h]
        psi : array [(km/s)²]  peculiar velocity correlation

        Unit check:
            [Hz_hMpc·fz]² × [Pt] × [dkt]
            = (km/s)²/(Mpc/h)² × (Mpc/h)³ × h/Mpc = (km/s)²  ✓
        """
        if r_grid is not None:
            r = np.asarray(r_grid)
        else:
            rmax = 2 * np.pi / kt.min() * 5
            r = np.logspace(-1, np.log10(rmax), 500)
        psi = np.zeros(len(r))
        for i, ri in enumerate(r):
            x = kt * ri
            j0 = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
            psi[i] = np.trapezoid(Pt * j0, kt)
        psi *= (a * Hz_hMpc * fz) ** 2 / (2 * np.pi**2)
        return r, psi

    def load_corrfunc_xi(self, xi_file, hdf5_file=None, rbins_file=None,
                         repo_root=None, nseeds=8, nthreads=4):
        """
        Load a Corrfunc xi(r) file and optionally estimate subsampling variance
        by re-running compute_xi with multiple random seeds.
        """
        cf = np.loadtxt(xi_file, comments='#')
        r_low = cf[:, 1]
        r_high = cf[:, 2]
        xi_cf = cf[:, 3]
        npairs = cf[:, 4]
        r_mid = np.sqrt(r_low * r_high) * self.h   # Mpc → Mpc/h

        xi_poisson_err = (1 + xi_cf) / np.sqrt(np.maximum(npairs, 1))
        xi_subsample_std = np.zeros_like(xi_cf)
        nseeds_used = 0

        xi_binary = os.path.join(repo_root, 'compute_xi') if repo_root else None
        if (nseeds > 1
                and xi_binary and os.path.exists(xi_binary)
                and rbins_file and os.path.exists(rbins_file)
                and hdf5_file and os.path.exists(hdf5_file)):
            xi_runs = []
            for seed in range(1, nseeds + 1):
                result = subprocess.run(
                    [xi_binary, hdf5_file, rbins_file, str(nthreads), '-s', str(seed)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = [ln for ln in result.stdout.splitlines() if not ln.startswith('#')]
                    vals = np.array([list(map(float, ln.split()))
                                     for ln in lines if ln.strip()])
                    if vals.ndim == 2 and vals.shape[0] == len(xi_cf):
                        xi_runs.append(vals[:, 3])
            if len(xi_runs) >= 2:
                xi_subsample_std = np.std(xi_runs, axis=0, ddof=1)
                nseeds_used = len(xi_runs)

        xi_err = np.sqrt(xi_poisson_err**2 + xi_subsample_std**2)
        self.corrfunc_xi = {
            "r_mid": r_mid, "xi": xi_cf, "err": xi_err, "nseeds_used": nseeds_used
        }

    def load_xi_cic(self, xi_cic_file, stem=None):
        """Load CIC density correlation ξ(r) from xi_cic_*.txt."""
        ngrid = None
        with open(xi_cic_file) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                m = re.search(r'Ngrid\s*:\s*(\d+)', line)
                if m:
                    ngrid = int(m.group(1))
        cic = np.loadtxt(xi_cic_file, comments='#')
        r_mid = np.sqrt(cic[:, 1] * cic[:, 2]) * self.h   # Mpc → Mpc/h
        self.xi_cic_list.append({
            "r_mid": r_mid, "xi": cic[:, 3],
            "stem": stem or os.path.basename(xi_cic_file),
            "ngrid": ngrid,
        })

    def load_vel_cic(self, vel_cic_file, a=1.0, stem=None):
        """
        Load CIC velocity correlation ψ(r) from vel_cic_*.txt.

        MUSIC2's SWIFT plugin writes Velocities in peculiar-velocity units
        (km/s), so the raw correlation from compute_xi_cic is already
        ψ_pec(r) in (km/s)². The `a` argument is kept for API compatibility
        but is unused.
        """
        del a  # SWIFT IC velocities are already v_pec; no rescaling needed
        vel = np.loadtxt(vel_cic_file, comments='#')
        r_mid = np.sqrt(vel[:, 1] * vel[:, 2]) * self.h   # Mpc → Mpc/h
        psi = vel[:, 3]
        self.vel_cic_list.append({
            "r_mid": r_mid, "psi": psi,
            "stem": stem or os.path.basename(vel_cic_file),
        })

    def auto_load_xi(self, nseeds=8, nthreads=4,
                     xi_cic_file=None, vel_cic_file=None):
        """
        Auto-detect and load xi/vel_cic/Corrfunc files for all pk runs.

        Looks alongside each pk_*.txt for:
          xi_<stem>.txt       → Corrfunc pair-counting ξ(r)  (single-run only)
          xi_cic_<stem>.txt   → CIC grid ξ(r)
          vel_cic_<stem>.txt  → CIC velocity ψ(r)

        Parameters
        ----------
        xi_cic_file, vel_cic_file : explicit paths for the primary run only
        """
        if not self.pk_runs:
            return

        repo_root = os.path.dirname(self.pk_runs[0]["data_dir"])

        # Corrfunc xi (single-run only)
        if len(self.pk_runs) == 1:
            run = self.pk_runs[0]
            xi_cf_path = os.path.join(run["data_dir"], f"xi_{run['stem']}.txt")
            if os.path.exists(xi_cf_path):
                hdf5_file = os.path.join(run["data_dir"], f"ics_swift_{run['stem']}.hdf5")
                rbins_file = os.path.join(run["data_dir"], f"rbins_{run['stem']}.txt")
                self.load_corrfunc_xi(
                    xi_cf_path, hdf5_file, rbins_file, repo_root,
                    nseeds=nseeds, nthreads=nthreads)

        # CIC xi and vel — load for every pk run
        for i, run in enumerate(self.pk_runs):
            stem = run["stem"]
            data_dir = run["data_dir"]
            z_run = run.get("z")
            a_run = 1.0 / (1.0 + z_run) if z_run is not None else 1.0

            # xi_cic: explicit file only for the primary (first) run
            if i == 0:
                path = xi_cic_file or os.path.join(data_dir, f"xi_cic_{stem}.txt")
            else:
                path = os.path.join(data_dir, f"xi_cic_{stem}.txt")
            if os.path.exists(path):
                self.load_xi_cic(path, stem=stem)

            # vel_cic: explicit file only for the primary run
            if i == 0:
                path = vel_cic_file or os.path.join(data_dir, f"vel_cic_{stem}.txt")
            else:
                path = os.path.join(data_dir, f"vel_cic_{stem}.txt")
            if os.path.exists(path):
                self.load_vel_cic(path, a=a_run, stem=stem)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot(self, show_nodeconv=False, hankel=False, show_shot_sub=None):
        """
        Create the four-panel P(k) / ξ(r)/ψ(r) figure with fractional residual
        panels below each main panel.

        Top-left    : P(k), theory, shot noise level, k_fund, k_Ny
        Bottom-left : (P_meas/P_theory − 1) fractional residual
        Top-right   : ξ(r) from theory / Corrfunc / CIC grid;
                      ψ(r) on a twin y-axis if velocity data loaded
        Bottom-right: (ξ_meas/ξ_theory − 1) fractional residual

        show_shot_sub : None (default) = auto: True when z ≤ 10, False at high z.
                        At low z, Poisson shot noise is a reasonable approximation
                        and the subtracted curve is useful.  At high z, IC particles
                        sit on a near-regular lattice (sub-Poissonian), so V/N
                        over-subtracts; only P_raw should be plotted there.
                        Pass True/False to override the auto behaviour.
        """
        # Auto-select show_shot_sub based on redshift of primary run
        if show_shot_sub is None:
            z = self.pk_runs[0].get("z") if self.pk_runs else None
            show_shot_sub = (z is not None and z <= 10)

        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({'font.size': 13})
        self.fig = plt.figure(figsize=(18, 7), constrained_layout=True)
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[3, 1.2], hspace=0.05)

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_pk_ratio = self.fig.add_subplot(gs[1, 0], sharex=self.ax)
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax_xi_ratio = self.fig.add_subplot(gs[1, 1], sharex=self.ax2)
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax_psi_ratio = self.fig.add_subplot(gs[1, 2], sharex=self.ax3)

        self._plot_pk_panel(self.ax, show_nodeconv=show_nodeconv, hankel=hankel,
                            show_shot_sub=show_shot_sub)
        self._plot_pk_ratio_panel(self.ax_pk_ratio, show_shot_sub=show_shot_sub)
        self._plot_xi_panel(self.ax2)
        self._plot_xi_ratio_panel(self.ax_xi_ratio)
        self._plot_psi_panel(self.ax3)
        self._plot_psi_ratio_panel(self.ax_psi_ratio)

        # Hide x tick labels on the main panels (shared with ratio panels below)
        for ax in (self.ax, self.ax2, self.ax3):
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel('')

    def _plot_pk_panel(self, ax, show_nodeconv=False, hankel=False, show_shot_sub=False):
        """Populate the left P(k) panel."""
        # Theory
        if self.theory_k is not None:
            z = self.pk_runs[0].get("z", "?") if self.pk_runs else "?"
            ax.loglog(self.theory_k, self.theory_P, 'k-', lw=1.2,
                      label=rf'Theory (CLASS, $z={z:.4g}$)', zorder=10)

        # Per-run curves
        multi = len(self.pk_runs) > 1
        for i, run in enumerate(self.pk_runs):
            k = run["k"]
            stem = run["stem"]
            color = f'C{i}'

            label = self._label_from_stem(stem) if multi else stem

            if show_nodeconv and i == 0:
                ax.loglog(k, run["Pk_nodeconv"], 's--', ms=3, lw=1.0,
                          color='C3', alpha=0.7, label='no CIC correction')

            ax.loglog(k, run["Pk_raw"], 'o-', ms=4, lw=1.2, color=color,
                      alpha=0.5, label=label)
            if show_shot_sub:
                pos = run["Pk_ss"] > 0
                ax.loglog(k[pos], run["Pk_ss"][pos], '^-', ms=4, lw=1.2,
                          color=color, alpha=0.5,
                          label=f'− shot noise ({label})' if multi else '− shot noise')

            if run["P_shot"] is not None:
                ax.axhline(run["P_shot"], color=color, ls='--', lw=0.8, alpha=0.6,
                           label=fr'$P_{{\rm shot}} = V/N = {run["P_shot"]:.2g}$ (Mpc/$h$)$^3$')

        # Reference lines: k_fund (dotted) and k_Ny (dashed) per run
        # Draw colored lines without labels; add two proxy legend entries.
        if self.pk_runs:
            import matplotlib.lines as mlines
            for i, run in enumerate(self.pk_runs):
                L = run["boxsize_mpch"]
                n = run["npart_side"] or 256
                if L:
                    kf = 2 * np.pi / L
                    knyq = np.pi * n / L
                    ax.axvline(kf,   color=f'C{i}', ls=':', lw=1.0)
                    ax.axvline(knyq, color=f'C{i}', ls='--', lw=1.0)
            # Proxy handles for k_fund / k_Ny — upper right
            ax.add_artist(ax.legend(
                handles=[
                    mlines.Line2D([], [], color='gray', ls=':',  lw=1.0, label=r'$k_{\rm fund}$'),
                    mlines.Line2D([], [], color='gray', ls='--', lw=1.0, label=r'$k_{\rm Ny}$'),
                ],
                fontsize='medium', loc='upper right'
            ))

        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
        if self.pk_runs:
            run = self.pk_runs[0]
            if len(self.pk_runs) == 1:
                z_str = "?" if run["z"] is None else f'{run["z"]:g}'
                ax.set_title(
                    f'N={run["npart_side"]}³, L={run["boxsize_mpch"]:.4g} Mpc/h, z={z_str}')
        ax.legend(fontsize='medium', loc='lower left')

        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        # 2π/[k_min, k_max] = [λ_max, λ_min]: already decreasing left→right,
        # matching k increasing left→right on the bottom axis.
        ax_top.set_xlim(2 * np.pi / np.array(ax.get_xlim()))
        ax_top.set_xlabel(r'$\lambda = 2\pi/k$ [Mpc/$h$]')

    def _plot_xi_panel(self, ax2):
        """Populate the right ξ(r) / ψ(r) panel."""
        multi = len(self.pk_runs) > 1

        # Theory ξ(r) — log y-axis; |ξ| with dashed for ξ<0
        if self.theory_xi_r is not None:
            z = self.pk_runs[0].get("z", "?") if self.pk_runs else "?"
            r = self.theory_xi_r
            xi_th = self.theory_xi
            pos = xi_th > 0
            neg = xi_th < 0
            ax2.plot(r[pos], xi_th[pos], 'k-', lw=1.2,
                     label=rf'Theory (CLASS, $z={z:.4g}$)', zorder=10)
            if neg.any():
                ax2.plot(r[neg], -xi_th[neg], 'k--', lw=1.2, zorder=10)

        # Corrfunc ξ(r) — |ξ| with solid (ξ>0) / dashed (ξ<0)
        if self.corrfunc_xi is not None:
            d = self.corrfunc_xi
            label = 'Corrfunc'
            if d["nseeds_used"] > 0:
                label += f' (±{d["nseeds_used"]}-seed)'
            r_c, xi_c, err_c = d["r_mid"], d["xi"], d["err"]
            pos = xi_c > 0
            neg = xi_c < 0
            ax2.errorbar(r_c[pos], xi_c[pos], yerr=err_c[pos],
                         fmt='s-', ms=4, lw=1.2, elinewidth=0.8, capsize=2,
                         color='C1', mfc='none', label=label)
            if neg.any():
                ax2.errorbar(r_c[neg], -xi_c[neg], yerr=err_c[neg],
                             fmt='s--', ms=4, lw=1.2, elinewidth=0.8, capsize=2,
                             color='C1', mfc='none')

        # CIC ξ(r) — |ξ| with solid (ξ>0) / dashed (ξ<0) segments
        for j, d in enumerate(self.xi_cic_list):
            color = f'C{j}'
            run_label = self._label_from_stem(d["stem"]) if multi else 'CIC grid'
            r_m, xi_m = d["r_mid"], d["xi"]
            pos = xi_m > 0
            neg = xi_m < 0
            ax2.plot(r_m[pos], xi_m[pos], 'o-', ms=4, lw=1.2,
                     color=color, alpha=0.5, label=run_label)
            if neg.any():
                ax2.plot(r_m[neg], -xi_m[neg], 'o--', ms=4, lw=1.2,
                         color=color, alpha=0.5)

        # Reference lines: L/2 (torus Nyquist for CIC-grid ξ) and Δx per run.
        # L/3 is drawn only when Corrfunc pair-count data is present (pair
        # counts beyond L/3 are biased by periodic images; the CIC FFT path
        # is exact on the torus up to L/2).
        if self.pk_runs:
            import matplotlib.lines as mlines
            show_L3 = self.corrfunc_xi is not None
            for i, run in enumerate(self.pk_runs):
                L = run["boxsize_mpch"]
                n = run["npart_side"]
                if L:
                    ax2.axvline(L / 2, color=f'C{i}', ls='--', lw=0.9)
                    if show_L3:
                        ax2.axvline(L / 3, color=f'C{i}', ls='-.', lw=0.9)
                if L and n:
                    ax2.axvline(L / n, color=f'C{i}', ls=':', lw=0.9)
            handles = [mlines.Line2D([], [], color='gray', ls='--', lw=1.0, label=r'$L/2$')]
            if show_L3:
                handles.append(
                    mlines.Line2D([], [], color='gray', ls='-.', lw=1.0, label=r'$L/3$'))
            handles.append(
                mlines.Line2D([], [], color='gray', ls=':', lw=1.0, label=r'$\Delta x$'))
            ax2.add_artist(ax2.legend(
                handles=handles, fontsize='medium', loc='upper right'))

        # Set xlim to span all loaded data
        all_r = [d["r_mid"] for d in self.xi_cic_list] + \
                [d["r_mid"] for d in self.vel_cic_list]
        if all_r:
            r_all = np.concatenate(all_r)
            ax2.set_xlim(r_all.min() * 0.8, r_all.max() * 1.2)

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$\xi(r)$')

        # y-range: driven by measured |ξ|; avoids the log-axis stretching down
        # to theory zero-crossings where |ξ| dips many decades below the signal.
        all_xi = [np.abs(d["xi"]) for d in self.xi_cic_list]
        if self.corrfunc_xi is not None:
            all_xi.append(np.abs(self.corrfunc_xi["xi"]))
        if all_xi:
            xi_abs = np.concatenate(all_xi)
            xi_abs = xi_abs[np.isfinite(xi_abs) & (xi_abs > 0)]
            if xi_abs.size:
                ymax = xi_abs.max() * 3.0
                ymin = ymax * 1e-5
                ax2.set_ylim(ymin, ymax)
        ax2.set_title(r'$\xi(r) = \langle \delta(x)\,\delta(x+r)\rangle$')

    def _plot_psi_panel(self, ax):
        """Populate the ψ(r) panel (third column, top row)."""
        multi = len(self.pk_runs) > 1

        # Theory ψ(r)
        if self.theory_psi is not None and self.pk_runs:
            L_max = max(r["boxsize_mpch"] for r in self.pk_runs if r["boxsize_mpch"])
            mask = ((self.theory_psi_r > 0) & (self.theory_psi > 0)
                    & (self.theory_psi_r < L_max / 2))
            z = self.pk_runs[0].get("z", "?")
            ax.loglog(self.theory_psi_r[mask], self.theory_psi[mask],
                      'k-', lw=1.2,
                      label=rf'Theory (CLASS, $z={z:.4g}$)', zorder=10)

        # Measured ψ(r) per run
        for j, d in enumerate(self.vel_cic_list):
            color = f'C{j}'
            run_label = self._label_from_stem(d["stem"]) if multi else 'CIC grid'
            pos = d["psi"] > 0
            ax.loglog(d["r_mid"][pos], d["psi"][pos], 'o-', ms=4, lw=1.2,
                      color=color, alpha=0.5, label=run_label)

        # Reference lines: L/2 per run
        if self.pk_runs:
            import matplotlib.lines as mlines
            for i, run in enumerate(self.pk_runs):
                L = run["boxsize_mpch"]
                if L:
                    ax.axvline(L / 2, color=f'C{i}', ls='--', lw=0.9)
            ax.add_artist(ax.legend(
                handles=[mlines.Line2D([], [], color='gray', ls='--', lw=1.0,
                                       label=r'$L/2$')],
                fontsize='medium', loc='upper right'
            ))

        # Match xlim to the ξ panel (same r-range as measured CIC ξ/ψ data)
        all_r = [d["r_mid"] for d in self.xi_cic_list] + \
                [d["r_mid"] for d in self.vel_cic_list]
        if all_r:
            r_all = np.concatenate(all_r)
            ax.set_xlim(r_all.min() * 0.8, r_all.max() * 1.2)

        ax.set_ylabel(r'$\psi(r)$ [(km/s)$^2$]')
        ax.set_title(r'$\psi(r) = \langle \mathbf{v}(x)\cdot \mathbf{v}(x+r)\rangle$')

    def _plot_psi_ratio_panel(self, ax):
        """ψ_meas / ψ_theory ratio panel (third column, bottom row)."""
        ax.axhline(1.0, color='k', lw=1.0)
        for level in [0.2, 0.4, 0.6, 0.8]:
            ax.axhline(level, color='gray', lw=0.5, ls='--', alpha=0.5)

        if self.theory_psi is not None:
            for j, d in enumerate(self.vel_cic_list):
                r = d["r_mid"]
                pos = d["psi"] > 0
                psi_th = np.interp(r[pos], self.theory_psi_r, self.theory_psi)
                valid = psi_th > 0
                r_v = r[pos][valid]
                ax.semilogx(r_v, d["psi"][pos][valid] / psi_th[valid],
                            'o-', ms=3, lw=1.2, color=f'C{j}', alpha=0.5)

        # Reference lines: L/2 per run
        for i, run in enumerate(self.pk_runs):
            L = run["boxsize_mpch"]
            if L:
                ax.axvline(L / 2, color=f'C{i}', ls='--', lw=0.9)

        ax.set_xlabel(r'$r$ [Mpc/$h$]')
        ax.set_ylabel('Measured / Theory', fontsize='medium')
        ax.set_ylim(0.0, 1.1)

    _RATIO_LEVELS = [0.01, 0.05, 0.10, 0.15]

    def _draw_ratio_reflines(self, ax, show_1pct_legend=False):
        """Draw ±5/10/15 % reference lines and a shaded ±1% band."""
        ax.axhline(0.0, color='k', lw=1.0)
        for level in self._RATIO_LEVELS:
            if level == 0.01:
                continue  # ±1% shown as shaded band instead
            ax.axhline(level, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.axhline(-level, color='gray', lw=0.5, ls='--', alpha=0.5)
        band = ax.axhspan(-0.01, 0.01, color='gray', alpha=0.2, lw=0,
                          label=r'$\pm 1\%$')
        if show_1pct_legend:
            ax.legend(handles=[band], loc='lower left', fontsize='medium',
                      framealpha=0.8)

    def _plot_pk_ratio_panel(self, ax, show_shot_sub=False):
        """Fractional residual (P_meas/P_theory − 1) below the P(k) panel."""
        self._draw_ratio_reflines(ax, show_1pct_legend=True)

        if self.theory_k is not None:
            for i, run in enumerate(self.pk_runs):
                k = run["k"]
                color = f'C{i}'
                Pt = np.interp(k, self.theory_k, self.theory_P)
                ax.semilogx(k, run["Pk_raw"] / Pt - 1.0,
                            'o-', ms=3, lw=1.2, color=color, alpha=0.5)
                if show_shot_sub:
                    pos = run["Pk_ss"] > 0
                    ax.semilogx(k[pos], run["Pk_ss"][pos] / Pt[pos] - 1.0,
                                '^-', ms=3, lw=1.0, color=color, alpha=0.5)

        # Repeat k_fund (dotted) / k_Ny (dashed) reference lines per run
        for i, run in enumerate(self.pk_runs):
            L = run["boxsize_mpch"]
            n = run["npart_side"] or 256
            if L:
                ax.axvline(2 * np.pi / L, color=f'C{i}', ls=':',  lw=1.0)
                ax.axvline(np.pi * n / L, color=f'C{i}', ls='--', lw=1.0)

        ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
        ax.set_ylabel('Measured/Theory - 1', fontsize='medium')
        ax.set_ylim(-0.16, 0.16)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:+.0%}'))

    def _plot_xi_ratio_panel(self, ax):
        """Fractional residual (ξ_meas/ξ_theory − 1) below the ξ(r) panel."""
        self._draw_ratio_reflines(ax)

        if self.theory_xi_r is not None:
            for j, d in enumerate(self.xi_cic_list):
                r = d["r_mid"]
                xi_th = np.interp(r, self.theory_xi_r, self.theory_xi)
                valid = xi_th != 0
                ax.semilogx(r[valid], d["xi"][valid] / xi_th[valid] - 1.0,
                            'o-', ms=3, lw=1.2, color=f'C{j}', alpha=0.5)

            if self.corrfunc_xi is not None:
                d = self.corrfunc_xi
                r = d["r_mid"]
                xi_th = np.interp(r, self.theory_xi_r, self.theory_xi)
                valid = xi_th != 0
                ax.semilogx(r[valid], d["xi"][valid] / xi_th[valid] - 1.0,
                            's-', ms=3, lw=1.2, color='C1', mfc='none', alpha=0.5)

        # Repeat L/3 and Δx reference lines for each run
        show_L3 = self.corrfunc_xi is not None
        for i, run in enumerate(self.pk_runs):
            L, n = run["boxsize_mpch"], run["npart_side"]
            if L:
                ax.axvline(L / 2, color=f'C{i}', ls='--', lw=0.9)
                if show_L3:
                    ax.axvline(L / 3, color=f'C{i}', ls='-.', lw=0.9)
            if L and n:
                ax.axvline(L / n, color=f'C{i}', ls=':', lw=0.9)

        ax.set_xlabel(r'$r$ [Mpc/$h$]')
        ax.set_ylabel('Measured/Theory - 1', fontsize='medium')
        ax.set_ylim(-0.16, 0.16)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:+.0%}'))

    def save(self, outfile, dpi=150):
        """Save the figure to a PNG file."""
        if self.fig is None:
            raise RuntimeError("Call plot() before save().")
        self.fig.savefig(outfile, dpi=dpi)
        print(f"Saved: {outfile}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot P(k) and xi(r)/psi(r) from saved pk_*.txt files.")
    parser.add_argument("pkfiles", nargs="+", help="One or more pk_*.txt files")
    parser.add_argument("--theory", default=None,
                        help="CLASS P(k) file (k[h/Mpc]  P[(Mpc/h)^3])")
    parser.add_argument("--theory-zref", type=float, default=None, metavar="ZREF",
                        help="If given, treat --theory as P(k) at ZREF and back-scale to "
                             "z_start using the no-radiation growth factor (matching MUSIC2's "
                             "ZeroRadiation=true). Use --theory-zref 0 with class_pk_z0_pk.dat "
                             "when validating high-z ICs (D+_norad(0)=1 makes this the natural "
                             "choice: P_ref = P_CLASS(k,0) × D+_norad(z_start)²).")
    parser.add_argument("--H0",      type=float, default=67.11)
    parser.add_argument("--Omega_m", type=float, default=0.3)
    parser.add_argument("--Omega_b", type=float, default=0.049)
    parser.add_argument("--xi-cic",  default=None, metavar="FILE",
                        help="Explicit xi_cic_*.txt (otherwise auto-detected)")
    parser.add_argument("--vel-cic", default=None, metavar="FILE",
                        help="Explicit vel_cic_*.txt (otherwise auto-detected)")
    parser.add_argument("--nseeds",  type=int, default=8,
                        help="Seeds for Corrfunc subsampling variance (default 8; 1=skip)")
    parser.add_argument("--show-nodeconv", action="store_true",
                        help="Also plot the un-deconvolved P(k) curve")
    parser.add_argument("--hankel",  action="store_true",
                        help="Overplot xi(r) from Hankel transform of measured P(k)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--show-shot-subtracted", dest="show_shot_sub",
                     action="store_true", default=None,
                     help="Force-show P(k) - V/N (overrides auto-detect).")
    grp.add_argument("--no-show-shot-subtracted", dest="show_shot_sub",
                     action="store_false",
                     help="Force-hide P(k) - V/N (overrides auto-detect). "
                          "Default: shown automatically when z ≤ 10.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG (default: pk_<stem>.png or pk_comparison.png)")
    args = parser.parse_args()

    plotter = ICPlotter(H0=args.H0, Omega_m=args.Omega_m, Omega_b=args.Omega_b)

    for pkfile in args.pkfiles:
        plotter.load_pk_file(pkfile)

    if args.theory:
        plotter.load_theory(args.theory, z_ref=args.theory_zref)

    plotter.auto_load_xi(
        nseeds=args.nseeds,
        xi_cic_file=args.xi_cic,
        vel_cic_file=args.vel_cic,
    )

    plotter.plot(show_nodeconv=args.show_nodeconv, hankel=args.hankel,
                 show_shot_sub=args.show_shot_sub)

    # Determine output filename
    if args.output:
        out_png = args.output
    elif len(args.pkfiles) == 1:
        stem = ICPlotter._stem_from_pkfile(args.pkfiles[0])
        data_dir = os.path.dirname(os.path.abspath(args.pkfiles[0]))
        plots_dir = os.path.join(os.path.dirname(data_dir), "plots")
        if os.path.isdir(plots_dir):
            out_png = os.path.join(plots_dir, f"pk_{stem}.png")
        else:
            out_png = f"pk_{stem}.png"
    else:
        out_png = "pk_comparison.png"

    plotter.save(out_png)


if __name__ == "__main__":
    main()
