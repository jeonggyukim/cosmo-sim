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
        self.h  = H0 / 100.0
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b

        # Loaded data
        self.pk_runs       = []     # list of dicts, one per pk_*.txt
        self.theory_k      = None   # CLASS k [h/Mpc]
        self.theory_P      = None   # CLASS P(k) [(Mpc/h)³]
        self.theory_xi_r   = None   # theory ξ(r) from Hankel transform
        self.theory_xi     = None
        self.theory_psi_r  = None   # theory ψ(r) [(km/s)²]
        self.theory_psi    = None
        self.corrfunc_xi   = None   # dict: r_mid, xi, err, nseeds_used
        self.xi_cic        = None   # dict: r_mid, xi
        self.vel_cic       = None   # dict: r_mid, psi

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
    def _parse_z_from_stem(stem):
        """Extract redshift from stem like 'n256_z2_L687' → 2.0."""
        m = re.search(r'_z([\d.]+)_', stem) or re.search(r'_z([\d.]+)$', stem)
        return float(m.group(1)) if m else None

    @staticmethod
    def _bao_scales(H0=67.11, Omega_m=0.3, Omega_b=0.049):
        h = H0 / 100.0
        Omh2 = Omega_m * h**2
        Obh2 = Omega_b * h**2
        r_d_Mpc  = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(3/4))
        r_d_mpch = r_d_Mpc * h
        return r_d_mpch, 2 * np.pi / r_d_mpch

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_pk_file(self, pkfile):
        """Load one pk_*.txt file (output of compute_pk.py)."""
        meta = self._parse_header(pkfile)
        data = np.loadtxt(pkfile, comments='#')
        # Columns: k  P_raw  P_shot_sub  P_nodeconv  sigma_P  nmodes
        run = {
            "pkfile":           pkfile,
            "stem":             self._stem_from_pkfile(pkfile),
            "data_dir":         os.path.dirname(os.path.abspath(pkfile)),
            "k":                data[:, 0],
            "Pk_raw":           data[:, 1],
            "Pk_ss":            data[:, 2],
            "Pk_nodeconv":      data[:, 3] if data.shape[1] > 3 else data[:, 1],
            "Pk_err":           data[:, 4] if data.shape[1] > 4 else np.zeros(len(data)),
            "P_shot":           meta["P_shot"],
            "boxsize_mpch":     meta["boxsize_mpch"],
            "N":                meta["N"],
            "npart_side":       round(meta["N"]**(1/3)) if meta["N"] else None,
            "z":                self._parse_z_from_stem(self._stem_from_pkfile(pkfile)),
        }
        self.pk_runs.append(run)

    def load_theory(self, theory_file):
        """
        Load CLASS P(k) and compute theory ξ(r) and ψ(r).

        ξ(r) is computed via Hankel transform using mcfit (if available).
        ψ(r) = [H(z)·f(z)]²/(2π²) ∫ P(k) j₀(kr) dk is computed via direct
        trapezoid integration.  Requires that at least one pk run has been
        loaded (to determine z).
        """
        kt, Pt = np.loadtxt(theory_file, comments='#', unpack=True)
        self.theory_k = kt
        self.theory_P = Pt

        # Theory ξ(r) via Hankel transform
        try:
            from mcfit import P2xi
            r, xi = P2xi(kt, l=0)(Pt, extrap=True)
            self.theory_xi_r = r
            self.theory_xi   = xi
        except ImportError:
            pass

        # Theory ψ(r): need z from the primary pk run
        if self.pk_runs:
            z = self.pk_runs[0].get("z")
            if z is not None:
                Ez = np.sqrt(self.Omega_m * (1+z)**3 + (1 - self.Omega_m))
                Hz        = self.H0 * Ez           # km/s/Mpc
                Hz_hMpc   = Hz / self.h            # km/s/(Mpc/h) — consistent with k in h/Mpc
                fz        = (self.Omega_m * (1+z)**3 / Ez**2) ** 0.55   # Linder 2005
                r_psi, psi = self._compute_theory_psi(kt, Pt, Hz_hMpc, fz)
                self.theory_psi_r = r_psi
                self.theory_psi   = psi
                print(f"Theory ψ(r): z={z:.4g}, H={Hz:.1f} km/s/Mpc, "
                      f"f={fz:.4f}, H·f/(h)={Hz_hMpc*fz:.2f} km/s/(Mpc/h)")

    @staticmethod
    def _compute_theory_psi(kt, Pt, Hz_hMpc, fz):
        """
        Compute the true peculiar-velocity correlation ψ(r) = ⟨v_pec(x)·v_pec(x+r)⟩.

        In linear theory the velocity field is irrotational:
            v_k = i (H·f/k) k̂ δ_k
        so the isotropic scalar correlation is:
            ψ_true(r) = [H·f]²/(2π²) ∫ P(k) j₀(kr) dk

        The k²/k² factor (from the velocity power spectrum vs the angular
        measure) cancels, unlike ξ(r) which has an extra k² in the integrand.

        Note on SWIFT velocity convention: SWIFT stores v_int = a × v_pec.
        The measured CIC correlation must be divided by a² before comparing
        to this theory curve (see load_vel_cic).

        Parameters
        ----------
        kt, Pt  : CLASS P(k): k in h/Mpc, P in (Mpc/h)³
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
        rmax = 2 * np.pi / kt.min() * 5
        r = np.logspace(-1, np.log10(rmax), 500)
        psi = np.zeros(len(r))
        for i, ri in enumerate(r):
            x  = kt * ri
            j0 = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
            psi[i] = np.trapezoid(Pt * j0, kt)
        psi *= (Hz_hMpc * fz) ** 2 / (2 * np.pi**2)
        return r, psi

    def load_corrfunc_xi(self, xi_file, hdf5_file=None, rbins_file=None,
                          repo_root=None, nseeds=8, nthreads=4):
        """
        Load a Corrfunc xi(r) file and optionally estimate subsampling variance
        by re-running compute_xi with multiple random seeds.
        """
        cf = np.loadtxt(xi_file, comments='#')
        r_low  = cf[:, 1];  r_high = cf[:, 2]
        xi_cf  = cf[:, 3];  npairs = cf[:, 4]
        r_mid  = np.sqrt(r_low * r_high) * self.h   # Mpc → Mpc/h

        xi_poisson_err    = (1 + xi_cf) / np.sqrt(np.maximum(npairs, 1))
        xi_subsample_std  = np.zeros_like(xi_cf)
        nseeds_used       = 0

        xi_binary = os.path.join(repo_root, 'compute_xi') if repo_root else None
        if (nseeds > 1
                and xi_binary and os.path.exists(xi_binary)
                and rbins_file and os.path.exists(rbins_file)
                and hdf5_file  and os.path.exists(hdf5_file)):
            xi_runs = []
            for seed in range(1, nseeds + 1):
                result = subprocess.run(
                    [xi_binary, hdf5_file, rbins_file, str(nthreads), '-s', str(seed)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = [l for l in result.stdout.splitlines() if not l.startswith('#')]
                    vals  = np.array([list(map(float, l.split()))
                                      for l in lines if l.strip()])
                    if vals.ndim == 2 and vals.shape[0] == len(xi_cf):
                        xi_runs.append(vals[:, 3])
            if len(xi_runs) >= 2:
                xi_subsample_std = np.std(xi_runs, axis=0, ddof=1)
                nseeds_used = len(xi_runs)

        xi_err = np.sqrt(xi_poisson_err**2 + xi_subsample_std**2)
        self.corrfunc_xi = {
            "r_mid": r_mid, "xi": xi_cf, "err": xi_err, "nseeds_used": nseeds_used
        }

    def load_xi_cic(self, xi_cic_file):
        """Load CIC density correlation ξ(r) from xi_cic_*.txt."""
        cic = np.loadtxt(xi_cic_file, comments='#')
        r_mid = np.sqrt(cic[:, 1] * cic[:, 2]) * self.h   # Mpc → Mpc/h
        self.xi_cic = {"r_mid": r_mid, "xi": cic[:, 3]}

    def load_vel_cic(self, vel_cic_file, a=1.0):
        """
        Load CIC velocity correlation ψ(r) from vel_cic_*.txt.

        SWIFT stores velocities as v_int = a × v_pec, so the raw correlation
        from compute_xi_cic is ψ_raw = a² × ψ_pec.  Pass a=1/(1+z) to
        convert to true peculiar-velocity units: ψ_pec = ψ_raw / a².
        """
        vel = np.loadtxt(vel_cic_file, comments='#')
        r_mid = np.sqrt(vel[:, 1] * vel[:, 2]) * self.h   # Mpc → Mpc/h
        psi   = vel[:, 3] / a**2                           # convert to v_pec units
        self.vel_cic = {"r_mid": r_mid, "psi": psi}

    def auto_load_xi(self, nseeds=8, nthreads=4,
                     xi_cic_file=None, vel_cic_file=None):
        """
        Auto-detect and load xi/vel_cic/Corrfunc files for the primary pk run.

        Looks alongside the first pk_*.txt for:
          xi_<stem>.txt       → Corrfunc pair-counting ξ(r)
          xi_cic_<stem>.txt   → CIC grid ξ(r)
          vel_cic_<stem>.txt  → CIC velocity ψ(r)

        Parameters
        ----------
        xi_cic_file, vel_cic_file : explicit paths (override auto-detection)
        """
        if not self.pk_runs:
            return
        run      = self.pk_runs[0]
        stem     = run["stem"]
        data_dir = run["data_dir"]
        repo_root = os.path.dirname(data_dir)

        # Corrfunc xi (single-run only)
        if len(self.pk_runs) == 1:
            xi_cf_path = os.path.join(data_dir, f"xi_{stem}.txt")
            if os.path.exists(xi_cf_path):
                hdf5_file  = os.path.join(data_dir, f"ics_swift_{stem}.hdf5")
                rbins_file = os.path.join(data_dir, f"rbins_{stem}.txt")
                self.load_corrfunc_xi(
                    xi_cf_path, hdf5_file, rbins_file, repo_root,
                    nseeds=nseeds, nthreads=nthreads)

        # CIC xi
        path = xi_cic_file or os.path.join(data_dir, f"xi_cic_{stem}.txt")
        if os.path.exists(path):
            self.load_xi_cic(path)

        # CIC vel — divide by a² to convert SWIFT v_int=a·v_pec → v_pec
        z_run = run.get("z")
        a_run = 1.0 / (1.0 + z_run) if z_run is not None else 1.0
        path = vel_cic_file or os.path.join(data_dir, f"vel_cic_{stem}.txt")
        if os.path.exists(path):
            self.load_vel_cic(path, a=a_run)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot(self, show_nodeconv=False, hankel=False, show_shot_sub=False):
        """
        Create the four-panel P(k) / ξ(r)/ψ(r) figure with fractional residual
        panels below each main panel.

        Top-left    : P(k), theory, shot noise level, k_fund, k_Ny
        Bottom-left : (P_meas/P_theory − 1) fractional residual
        Top-right   : ξ(r) from theory / Corrfunc / CIC grid;
                      ψ(r) on a twin y-axis if velocity data loaded
        Bottom-right: (ξ_meas/ξ_theory − 1) fractional residual

        show_shot_sub : if True, also plot P_raw - V/N (shot-noise-subtracted).
                        Off by default: IC particles sit on a near-regular lattice,
                        which is sub-Poissonian, so V/N over-estimates the true
                        discreteness noise.  The raw P(k) with V/N shown as a
                        reference line is the correct display for lattice ICs.
        """
        from matplotlib.gridspec import GridSpec
        self.fig = plt.figure(figsize=(13, 8))
        gs = GridSpec(2, 2, height_ratios=[4, 1], hspace=0.05, wspace=0.32,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

        self.ax         = self.fig.add_subplot(gs[0, 0])
        self.ax_pk_ratio = self.fig.add_subplot(gs[1, 0], sharex=self.ax)
        self.ax2        = self.fig.add_subplot(gs[0, 1])
        self.ax_xi_ratio = self.fig.add_subplot(gs[1, 1], sharex=self.ax2)

        self._plot_pk_panel(self.ax, show_nodeconv=show_nodeconv, hankel=hankel,
                            show_shot_sub=show_shot_sub)
        self._plot_pk_ratio_panel(self.ax_pk_ratio, show_shot_sub=show_shot_sub)
        self._plot_xi_panel(self.ax2)
        self._plot_xi_ratio_panel(self.ax_xi_ratio)

        # Hide x tick labels on the main panels (shared with ratio panels below)
        plt.setp(self.ax.get_xticklabels(), visible=False)
        self.ax.set_xlabel('')
        plt.setp(self.ax2.get_xticklabels(), visible=False)
        self.ax2.set_xlabel('')

    def _plot_pk_panel(self, ax, show_nodeconv=False, hankel=False, show_shot_sub=False):
        """Populate the left P(k) panel."""
        # Theory
        if self.theory_k is not None:
            ax.loglog(self.theory_k, self.theory_P, 'k-', lw=1.2,
                      label='theory (CLASS)', zorder=10)

        # Per-run curves
        for i, run in enumerate(self.pk_runs):
            k      = run["k"]
            label  = run["stem"]
            color  = f'C{i}'

            if show_nodeconv and i == 0:
                ax.loglog(k, run["Pk_nodeconv"], 's--', ms=3, lw=1.0,
                          color='C3', alpha=0.7, label='no CIC correction')

            ax.loglog(k, run["Pk_raw"], 'o-', ms=4, lw=1.2, color=color,
                      label=f'CIC-corrected ({label})')
            if show_shot_sub:
                pos = run["Pk_ss"] > 0
                ax.loglog(k[pos], run["Pk_ss"][pos], '^-', ms=4, lw=1.2,
                          color=color, alpha=0.6,
                          label=f'− shot noise ({label})')

            if run["P_shot"] is not None:
                ax.axhline(run["P_shot"], color=color, ls='--', lw=0.8, alpha=0.6,
                           label=fr'$P_{{\rm shot}} = V/N = {run["P_shot"]:.2g}$ (Mpc/$h$)$^3$')

        # Reference lines from primary run
        if self.pk_runs:
            run = self.pk_runs[0]
            L   = run["boxsize_mpch"]
            n   = run["npart_side"] or 256
            if L:
                kf   = 2 * np.pi / L
                knyq = np.pi * n / L
                ax.axvline(kf,   color='C2',   ls=':', lw=1.0,
                           label=fr'$k_{{\rm fund}}$ = {kf:.2g} $h$/Mpc')
                ax.axvline(knyq, color='gray', ls='--', lw=1.0,
                           label=fr'$k_{{\rm Ny}}$ = {knyq:.2g} $h$/Mpc')

        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
        if self.pk_runs:
            run = self.pk_runs[0]
            ax.set_title(f'N={run["npart_side"]}³, L={run["boxsize_mpch"]:.4g} Mpc/h, '
                         f'z={run["z"] or "?"}')
        ax.legend(fontsize="medium")

        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        # 2π/[k_min, k_max] = [λ_max, λ_min]: already decreasing left→right,
        # matching k increasing left→right on the bottom axis.
        ax_top.set_xlim(2 * np.pi / np.array(ax.get_xlim()))
        ax_top.set_xlabel(r'$\lambda = 2\pi/k$ [Mpc/$h$]')

    def _plot_xi_panel(self, ax2):
        """Populate the right ξ(r) / ψ(r) panel."""
        # Theory ξ(r)
        if self.theory_xi_r is not None:
            mask = (self.theory_xi_r > 0) & (self.theory_xi > 0)
            ax2.loglog(self.theory_xi_r[mask], self.theory_xi[mask],
                       'k-', lw=1.2, label='theory ξ (CLASS)', zorder=10)

        # Corrfunc xi(r)
        if self.corrfunc_xi is not None:
            d = self.corrfunc_xi
            pos = d["xi"] > 0
            if pos.sum() >= len(d["xi"]) / 2:
                label = 'measured ξ (Corrfunc)'
                if d["nseeds_used"] > 0:
                    label += f' ±{d["nseeds_used"]}-seed'
                ax2.errorbar(d["r_mid"][pos], d["xi"][pos], yerr=d["err"][pos],
                             fmt='s-', ms=4, lw=1.2, elinewidth=0.8, capsize=2,
                             color='C1', mfc='none', label=label)

        # CIC xi(r)
        if self.xi_cic is not None:
            d = self.xi_cic
            pos, neg = d["xi"] > 0, d["xi"] < 0
            ax2.loglog(d["r_mid"][pos], d["xi"][pos], '^-', ms=4, lw=1.2,
                       color='C4', label=r'measured ξ (CIC grid, $>0$)')
            if neg.any():
                ax2.loglog(d["r_mid"][neg], np.abs(d["xi"][neg]), '^--',
                           ms=4, lw=0.8, color='C4', alpha=0.4,
                           label=r'measured ξ (CIC grid, $<0$)')

        # ψ(r) on twin y-axis
        has_psi = (self.vel_cic is not None) or (self.theory_psi is not None)
        if has_psi:
            ax3 = ax2.twinx()
            ax3.set_ylabel(r'$\psi(r)$ [$v_\mathrm{pec}$, (km/s)$^2$]', color='C5')
            ax3.tick_params(axis='y', labelcolor='C5')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            self.ax3 = ax3

            if self.vel_cic is not None:
                d   = self.vel_cic
                pos = d["psi"] > 0
                ax3.plot(d["r_mid"][pos], d["psi"][pos], 'D-', ms=4, lw=1.2,
                         color='C5', label=r'measured $\psi(r)$ (CIC grid)')

            if self.theory_psi is not None and self.pk_runs:
                L = self.pk_runs[0]["boxsize_mpch"]
                mask = ((self.theory_psi_r > 0) & (self.theory_psi > 0)
                        & (self.theory_psi_r < (L / 2 if L else np.inf)))
                z = self.pk_runs[0].get("z", "?")
                ax3.loglog(self.theory_psi_r[mask], self.theory_psi[mask],
                           'k--', lw=1.2,
                           label=rf'theory $\psi(r)$ (CLASS, $z={z:.4g}$)')

            # Merge legends from both axes
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax2.legend(lines2 + lines3, labels2 + labels3,
                       fontsize="medium", loc='lower left')
        else:
            ax2.legend(fontsize="medium", loc='lower left')

        # Reference lines
        if self.pk_runs:
            run = self.pk_runs[0]
            L   = run["boxsize_mpch"]
            n   = run["npart_side"]
            if L:
                ax2.axvline(L / 3, color='gray', ls='--', lw=1.0,
                            label=fr'$L/3$ = {L/3:.4g} Mpc/$h$')
            if L and n:
                dx = L / n
                ax2.axvline(dx, color='C2', ls=':', lw=1.0,
                            label=fr'$\Delta x$ = {dx:.2g} Mpc/$h$')

        ax2.set_ylabel(r'$\xi(r)$')
        ax2.set_title(r'Correlation functions $\xi(r)$ and $\psi(r)$')

    _RATIO_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    def _draw_ratio_reflines(self, ax):
        """Draw ±1/5/10/20/30/40/50 % horizontal reference lines."""
        ax.axhline(0.0, color='k', lw=1.0)
        for level in self._RATIO_LEVELS:
            ax.axhline( level, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.axhline(-level, color='gray', lw=0.5, ls='--', alpha=0.5)

    def _plot_pk_ratio_panel(self, ax, show_shot_sub=False):
        """Fractional residual (P_meas/P_theory − 1) below the P(k) panel."""
        self._draw_ratio_reflines(ax)

        if self.theory_k is not None:
            for i, run in enumerate(self.pk_runs):
                k     = run["k"]
                color = f'C{i}'
                Pt    = np.interp(k, self.theory_k, self.theory_P)
                ax.semilogx(k, run["Pk_raw"] / Pt - 1.0,
                            'o-', ms=3, lw=1.2, color=color)
                if show_shot_sub:
                    pos = run["Pk_ss"] > 0
                    ax.semilogx(k[pos], run["Pk_ss"][pos] / Pt[pos] - 1.0,
                                '^-', ms=3, lw=1.0, color=color, alpha=0.6)

        # Repeat k_fund / k_Ny reference lines
        if self.pk_runs:
            run = self.pk_runs[0]
            L, n = run["boxsize_mpch"], run["npart_side"] or 256
            if L:
                ax.axvline(2 * np.pi / L,    color='C2',   ls=':', lw=1.0)
                ax.axvline(np.pi * n / L,    color='gray', ls='--', lw=1.0)

        ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
        ax.set_ylabel(r'$P_{\rm meas}/P_{\rm theory} - 1$', fontsize=9)
        ax.set_ylim(-0.55, 0.55)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:+.0%}'))

    def _plot_xi_ratio_panel(self, ax):
        """Fractional residual (ξ_meas/ξ_theory − 1) below the ξ(r) panel."""
        self._draw_ratio_reflines(ax)

        if self.theory_xi_r is not None:
            if self.xi_cic is not None:
                d   = self.xi_cic
                pos = d["xi"] > 0
                r   = d["r_mid"][pos]
                xi_th = np.interp(r, self.theory_xi_r, self.theory_xi)
                valid = xi_th > 0
                ax.semilogx(r[valid], d["xi"][pos][valid] / xi_th[valid] - 1.0,
                            '^-', ms=3, lw=1.2, color='C4')

            if self.corrfunc_xi is not None:
                d   = self.corrfunc_xi
                pos = d["xi"] > 0
                r   = d["r_mid"][pos]
                xi_th = np.interp(r, self.theory_xi_r, self.theory_xi)
                valid = xi_th > 0
                ax.semilogx(r[valid], d["xi"][pos][valid] / xi_th[valid] - 1.0,
                            's-', ms=3, lw=1.2, color='C1', mfc='none')

        # Repeat L/3 and Δx reference lines
        if self.pk_runs:
            run = self.pk_runs[0]
            L, n = run["boxsize_mpch"], run["npart_side"]
            if L:
                ax.axvline(L / 3, color='gray', ls='--', lw=1.0)
            if L and n:
                ax.axvline(L / n, color='C2',   ls=':', lw=1.0)

        ax.set_xlabel(r'$r$ [Mpc/$h$]')
        ax.set_ylabel(r'$\xi_{\rm meas}/\xi_{\rm theory} - 1$', fontsize=9)
        ax.set_ylim(-0.55, 0.55)
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
    parser.add_argument("--show-shot-subtracted", action="store_true",
                        help="Also plot P(k) - V/N (Poisson shot-noise subtracted). "
                             "Off by default: IC particles sit on a near-regular lattice "
                             "(sub-Poissonian), so V/N over-subtracts.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG (default: pk_<stem>.png or pk_comparison.png)")
    args = parser.parse_args()

    plotter = ICPlotter(H0=args.H0, Omega_m=args.Omega_m, Omega_b=args.Omega_b)

    for pkfile in args.pkfiles:
        plotter.load_pk_file(pkfile)

    if args.theory:
        plotter.load_theory(args.theory)

    plotter.auto_load_xi(
        nseeds=args.nseeds,
        xi_cic_file=args.xi_cic,
        vel_cic_file=args.vel_cic,
    )

    plotter.plot(show_nodeconv=args.show_nodeconv, hankel=args.hankel,
                 show_shot_sub=args.show_shot_subtracted)

    # Determine output filename
    if args.output:
        out_png = args.output
    elif len(args.pkfiles) == 1:
        stem     = ICPlotter._stem_from_pkfile(args.pkfiles[0])
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
