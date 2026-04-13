#!/usr/bin/env python3
"""
plot_pk.py — Re-plot P(k) (and xi(r)) from saved pk_*.txt files.

Usage:
    # Single run
    python plot_pk.py data/pk_n256_z45_L500.txt

    # Multiple runs overlaid
    python plot_pk.py data/pk_n256_z45_L500.txt data/pk_n512_z45_L500.txt

    # With CLASS theory
    python plot_pk.py data/pk_n256_z45_L500.txt --theory data/class_pk_z45_pk.dat

    # Custom output
    python plot_pk.py data/pk_*.txt --theory data/class_pk_z45_pk.dat -o comparison.png

Output: PNG with left panel P(k), right panel xi(r) (Hankel transform of measured P(k)).
"""

import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_header(path):
    """Extract BoxSize [Mpc/h], N, P_shot from the txt header comments."""
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


# ---------------------------------------------------------------------------
# xi(r) via direct integration of measured P(k)
# ---------------------------------------------------------------------------

def pk_to_xi(k, Pk, boxsize_mpch):
    """
    xi(r) = 1/(2pi^2) int P(k) sinc(kr) k^2 dk
    Integrates over the measured k bins (no Hankel transform library needed).
    """
    r = np.logspace(-1, np.log10(boxsize_mpch / 2), 300)
    xi = np.zeros(len(r))
    for i, ri in enumerate(r):
        x = k * ri
        sinc = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
        xi[i] = np.trapezoid(Pk * sinc * k**2, k) / (2 * np.pi**2)
    return r, xi


# ---------------------------------------------------------------------------
# BAO scale (Eisenstein & Hu 1998)
# ---------------------------------------------------------------------------

def bao_scales(H0=67.11, Omega_m=0.3, Omega_b=0.049):
    h = H0 / 100.0
    Omh2 = Omega_m * h**2
    Obh2 = Omega_b * h**2
    r_d_Mpc  = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(3/4))
    r_d_mpch = r_d_Mpc * h
    k_BAO    = 2 * np.pi / r_d_mpch
    return r_d_mpch, k_BAO


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-plot P(k) from saved pk_*.txt files.")
    parser.add_argument("pkfiles", nargs="+", help="One or more pk_*.txt files")
    parser.add_argument("--theory", default=None,
                        help="CLASS P(k) ASCII file (columns: k[h/Mpc]  P[(Mpc/h)^3])")
    parser.add_argument("--H0", type=float, default=67.11)
    parser.add_argument("--Omega_m", type=float, default=0.3)
    parser.add_argument("--Omega_b", type=float, default=0.049)
    parser.add_argument("--no-xi", action="store_true",
                        help="Skip xi(r) panel (faster)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG (default: pk_comparison.png)")
    args = parser.parse_args()

    r_d_mpch, k_BAO = bao_scales(args.H0, args.Omega_m, args.Omega_b)

    # --- Load theory ---
    kt = Pt = None
    if args.theory:
        kt, Pt = np.loadtxt(args.theory, comments="#", unpack=True)

    # --- Set up figure ---
    ncols = 1 if args.no_xi else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    ax = axes[0]
    ax2 = axes[1] if not args.no_xi else None

    # --- Theory curves ---
    if kt is not None:
        ax.loglog(kt, Pt, "k-", lw=1.2, label="theory (CLASS)", zorder=10)
        if ax2 is not None:
            try:
                from mcfit import P2xi
                r_th, xi_th = P2xi(kt, l=0)(Pt, extrap=True)
                mask = (r_th > 0) & (xi_th > 0)
                ax2.loglog(r_th[mask], xi_th[mask], "k-", lw=1.2,
                           label="theory (CLASS)", zorder=10)
            except ImportError:
                pass  # mcfit optional

    # --- Per-file curves ---
    for i, path in enumerate(args.pkfiles):
        meta = parse_header(path)
        data = np.loadtxt(path, comments="#")
        k_cen      = data[:, 0]   # h/Mpc
        Pk_raw     = data[:, 1]   # (Mpc/h)^3, no shot subtraction
        P_shot     = meta["P_shot"]
        boxsize    = meta["boxsize_mpch"]
        N          = meta["N"]
        npart_side = round(N ** (1.0 / 3.0)) if N else None

        label = os.path.basename(path).replace("pk_", "").replace(".txt", "")

        ax.loglog(k_cen, Pk_raw, "o-", ms=3, lw=1.0, color=f"C{i}", label=label)
        if P_shot is not None:
            ax.axhline(P_shot, color=f"C{i}", ls="--", lw=0.8, alpha=0.6)

        if ax2 is not None and boxsize is not None:
            r_xi, xi_meas = pk_to_xi(k_cen, Pk_raw, boxsize)
            pos = xi_meas > 0
            ax2.loglog(r_xi[pos], xi_meas[pos], "o-", ms=3, lw=1.0,
                       color=f"C{i}", alpha=0.7, label=label)

    # --- Reference lines on P(k) panel ---
    ax.axvline(k_BAO, color="purple", ls="-", lw=1.0,
               label=fr"$k_\mathrm{{BAO}}$ = {k_BAO:.3f} $h$/Mpc")
    ax.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/$h$)$^3$]")
    ax.set_title(r"Matter power spectrum $P(k)$")
    ax.legend(fontsize=8)

    ax_top = ax.twiny()
    ax_top.set_xscale("log")
    ax_top.set_xlim(2 * np.pi / np.array(ax.get_xlim()))
    ax_top.invert_xaxis()
    ax_top.set_xlabel(r"$\lambda = 2\pi/k$ [Mpc/$h$]")

    # --- Reference lines on xi(r) panel ---
    if ax2 is not None:
        ax2.axvline(r_d_mpch, color="purple", ls="-", lw=1.0,
                    label=fr"$r_d$ = {r_d_mpch:.1f} Mpc/$h$")
        ax2.set_xlabel(r"$r$ [Mpc/$h$]")
        ax2.set_ylabel(r"$\xi(r)$")
        ax2.set_title(r"Correlation function $\xi(r)$")
        ax2.legend(fontsize=8)

    fig.tight_layout()

    out = args.output or "pk_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
