#!/usr/bin/env python3
"""plot_box_size_comparison.py — Density vs velocity diagnostics across box sizes.

Generates a 4-panel figure that demonstrates the central physics point of
the velocity-power-spectrum diagnostic: the velocity field is much more
sensitive to long-wavelength modes than the density field, so smaller
simulation boxes (which truncate modes below k_fund = 2π/L) underestimate
the velocity statistics far more than the density ones.

Layout:
                            k-space            r-space
        density       P_delta(k) overlay   xi(r) overlay
        velocity      P_v(k)     overlay   psi(r) overlay

Each panel overlays measurements from several IC box sizes against a
single linear-theory prediction (CLASS P(k) at the IC redshift).

Inputs (defaults expect outputs from the standard pipeline at z=200):
    data/pk_n256_z200_L{256,512,1024}.txt
    data/pv_n256_z200_L{256,512,1024}.txt
    data/xi_cic_n256_z200_L{256,512,1024}.txt
    data/vel_cic_n256_z200_L{256,512,1024}.txt
    data/class_pk_z200_pk.dat

Usage:
    conda run -n cosmo python scripts/analysis/plot_box_size_comparison.py
    # output: plots/box_size_comparison.png

References:
- Klypin & Prada 2019, MNRAS 489, 1684 — general finite-box effects.
- Chuang+ 2026, arXiv:2602.04485 — velocity-correlation suppression
  for small boxes, with k_min marginalisation as remedy.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from icpipe import LinearTheory
from icpipe.io import read_pk, read_pv


# ---------------------------------------------------------------------------
# Helpers to load the existing xi_cic / vel_cic ASCII tables
# ---------------------------------------------------------------------------

def read_xi_cic(path: str, h: float = 0.6711) -> tuple[np.ndarray, np.ndarray]:
    """Read xi_cic_*.txt: r in Mpc, xi (dimensionless). Convert r to Mpc/h."""
    arr = np.loadtxt(path, comments="#")
    r_mid_mpc = np.sqrt(arr[:, 1] * arr[:, 2])  # geometric mean of bin edges
    xi = arr[:, 3]
    return r_mid_mpc * h, xi


def read_vel_cic(path: str, h: float = 0.6711) -> tuple[np.ndarray, np.ndarray]:
    """Read vel_cic_*.txt: r in Mpc, psi in (km/s)^2. Convert r to Mpc/h."""
    arr = np.loadtxt(path, comments="#")
    r_mid_mpc = np.sqrt(arr[:, 1] * arr[:, 2])
    psi = arr[:, 3]
    return r_mid_mpc * h, psi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--boxes", type=int, nargs="+", default=[256, 512, 1024],
                   help="Box sizes (Mpc/h) to overlay (default: 256 512 1024)")
    p.add_argument("--ngrid", type=int, default=256,
                   help="Particle grid size N (default: 256, used in filenames)")
    p.add_argument("--z", type=int, default=200,
                   help="IC redshift (default: 200, used in filenames)")
    p.add_argument("--data-dir", default="data",
                   help="Where the pk/pv/xi/vel files live")
    p.add_argument("--class-pk", default="data/class_pk_z200_pk.dat",
                   help="CLASS P(k) file for theory curves")
    p.add_argument("--H0", type=float, default=67.11,
                   help="H0 in km/s/Mpc")
    p.add_argument("--Omega-m", type=float, default=0.3,
                   help="Omega_m today (matches conf/CV_22_MUSIC_template.conf)")
    p.add_argument("-o", "--output", default="plots/box_size_comparison.png",
                   help="Output PNG path (default: plots/box_size_comparison.png)")
    args = p.parse_args()

    h = args.H0 / 100.0
    th = LinearTheory.from_class(args.class_pk, z=args.z, h=h, Omega_m=args.Omega_m)

    print(f"Theory: z={args.z}, h={h}, Omega_m={args.Omega_m}")
    print(f"        a={th.a:.3e}, H={th.H_kmsMpc:.3g} km/s/Mpc, "
          f"f={th.f_growth:.4g}, aHf={th.aHf:.3g} km/s/Mpc")

    # ----- load measurements per box ----------------------------------
    runs = {}
    for L in args.boxes:
        stem = f"n{args.ngrid}_z{args.z}_L{L}"
        pk_file = os.path.join(args.data_dir, f"pk_{stem}.txt")
        pv_file = os.path.join(args.data_dir, f"pv_{stem}.txt")
        xi_file = os.path.join(args.data_dir, f"xi_cic_{stem}.txt")
        vel_file = os.path.join(args.data_dir, f"vel_cic_{stem}.txt")

        d = {"L": L, "kfund": 2 * np.pi / L}
        if os.path.exists(pk_file):
            d["pk"] = read_pk(pk_file)
        if os.path.exists(pv_file):
            d["pv"] = read_pv(pv_file)
        if os.path.exists(xi_file):
            r, xi = read_xi_cic(xi_file, h=h)
            d["xi"] = (r, xi)
        if os.path.exists(vel_file):
            r, psi = read_vel_cic(vel_file, h=h)
            d["vel"] = (r, psi)
        runs[L] = d
        print(f"L = {L:4d} Mpc/h: pk={'pk' in d}  pv={'pv' in d}  "
              f"xi={'xi' in d}  vel={'vel' in d}  k_fund={d['kfund']:.3g}")

    # ----- theory grids -----------------------------------------------
    k_th = np.logspace(np.log10(min(2*np.pi/L for L in args.boxes) * 0.5),
                       1.0, 400)
    Pk_th = th.Pk(k_th)
    Pv_th = th.Pv(k_th)

    rmin = min(min(d["xi"][0].min() for d in runs.values() if "xi" in d),
               min(d["vel"][0].min() for d in runs.values() if "vel" in d))
    rmax = max(max(d["xi"][0].max() for d in runs.values() if "xi" in d),
               max(d["vel"][0].max() for d in runs.values() if "vel" in d))
    r_th = np.logspace(np.log10(max(rmin, 0.5)), np.log10(rmax * 1.2), 200)
    xi_th = th.xi(r_th)
    psi_th = th.psi(r_th)

    # ----- figure ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                              constrained_layout=True)

    colors = ["C0", "C1", "C2", "C3"]

    # P_delta(k)
    ax = axes[0, 0]
    ax.loglog(k_th, Pk_th, 'k-', lw=1.2, label="linear theory")
    for L, c in zip(args.boxes, colors):
        d = runs[L]
        if "pk" not in d:
            continue
        ax.loglog(d["pk"]["k"], d["pk"]["P_raw"], 'o-', color=c, ms=3, lw=1,
                  label=fr"$L={L}\,h^{{-1}}$Mpc")
        ax.axvline(d["kfund"], color=c, ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_\delta(k)\;[(h^{-1}\mathrm{Mpc})^3]$")
    ax.set_title(r"density $P_\delta(k)$  (k-space)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # P_v(k)
    ax = axes[1, 0]
    ax.loglog(k_th, Pv_th, 'k-', lw=1.2, label="linear theory")
    for L, c in zip(args.boxes, colors):
        d = runs[L]
        if "pv" not in d:
            continue
        ax.loglog(d["pv"]["k"], d["pv"]["Pv_raw"], 's-', color=c, ms=3, lw=1,
                  label=fr"$L={L}\,h^{{-1}}$Mpc")
        ax.axvline(d["kfund"], color=c, ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_v(k)\;[(\mathrm{km/s})^2(h^{-1}\mathrm{Mpc})^3]$")
    ax.set_title(r"velocity $P_v(k)$  (k-space)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # xi(r)
    ax = axes[0, 1]
    ax.semilogx(r_th, r_th**2 * xi_th, 'k-', lw=1.2, label="linear theory")
    for L, c in zip(args.boxes, colors):
        d = runs[L]
        if "xi" not in d:
            continue
        r, xi = d["xi"]
        ax.semilogx(r, r**2 * xi, 'o-', color=c, ms=3, lw=1,
                    label=fr"$L={L}\,h^{{-1}}$Mpc")
    ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$r^2\,\xi(r)$")
    ax.set_title(r"density $\xi(r)$  (r-space)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # psi(r)
    ax = axes[1, 1]
    ax.semilogx(r_th, psi_th, 'k-', lw=1.2, label="linear theory")
    for L, c in zip(args.boxes, colors):
        d = runs[L]
        if "vel" not in d:
            continue
        r, psi = d["vel"]
        ax.semilogx(r, psi, 's-', color=c, ms=3, lw=1,
                    label=fr"$L={L}\,h^{{-1}}$Mpc")
    ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\psi(r)\;[(\mathrm{km/s})^2]$")
    ax.set_title(r"velocity $\psi(r)$  (r-space)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(rf"IC validation across box sizes: $z={args.z}$, $N={args.ngrid}^3$ "
                 r"(dotted: $k_{\rm fund}=2\pi/L$ per box)",
                 fontsize=12)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
