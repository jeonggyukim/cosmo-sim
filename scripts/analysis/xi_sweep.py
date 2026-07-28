#!/usr/bin/env python3
"""xi_sweep.py — measure xi(r) and P(k) over many random seeds, then overplot.

Two subcommands, deliberately separate, because the measuring is the
expensive half and the plotting is the half you re-run twenty times while
fiddling with line styles:

    compute   one seed -> one .npz file.  Skips seeds already on disk, so the
              sweep is resumable and can be grown by re-running with a wider
              --seeds range.
    plot      load every .npz in a directory and overlay them.

Only the padded-FFT estimator runs during a sweep. The direct lag sum costs
O(k_max^3 N_cell) per seed against O(N log N) for the FFT, and the two are
the same estimator evaluated two ways (see notes/xi_estimators.pdf). Pass
--check-identity to also run the direct sum and record the worst per-bin
relative difference in the .npz; useful on one or two seeds as a spot check,
wasteful on all of them.

Geometry is a pencil beam: a column cut from a periodic box, narrow in x and
y, spanning the box in z. The cut axes are zero-padded to 2N so their
autocorrelation is linear; the long axis is left unpadded because there the
wrap-around pairs are real.

Output format is .npz rather than text because a run produces several arrays
(r, xi, k, two P(k)s) plus scalar metadata, and .npz stores them under names
with no parser and no lost digits. Pass --also-txt to additionally write the
column format that the HR-ICs-pipeline plotting scripts read.

Usage:
    # 16 seeds, resumable
    conda run -n cosmo python scripts/analysis/xi_sweep.py compute \\
        --seeds 1000 1015 --outdir data/xi_sweep

    # spot-check the estimator identity on one seed
    conda run -n cosmo python scripts/analysis/xi_sweep.py compute \\
        --seeds 1000 1000 --outdir data/xi_sweep --check-identity --force

    # overlay everything found
    conda run -n cosmo python scripts/analysis/xi_sweep.py plot \\
        --outdir data/xi_sweep --output plots/xi_sweep.png
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The estimators and the field generator live with the note's figure scripts,
# so there is exactly one implementation of each.
_FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "notes", "figures", "xi_estimators")
sys.path.insert(0, os.path.normpath(_FIGDIR))

import plot_xi_identity as xid   # noqa: E402


# --------------------------------------------------------------- compute ----
def measure_one(seed, rmin, rmax, nbins, check_identity):
    """Run one seed. Returns a dict ready for np.savez."""
    t0 = time.time()

    delta_full = xid.make_field(seed)
    i0 = int(xid.PENCIL[0] * xid.NFULL)
    i1 = int(xid.PENCIL[1] * xid.NFULL)
    delta = delta_full[i0:i1, i0:i1, :].copy()
    delta -= delta.mean()
    Nx, Ny, Nz = delta.shape
    cells = (xid.CELL,) * 3

    lags = xid.lag_list((Nx, Ny, Nz), cells, rmin, rmax)
    num = xid.s_padded_fft(delta)
    xi_lag = np.array([num[di, dj, dk] / npair for di, dj, dk, _, npair in lags])

    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r, xi, nlag = xid.bin_xi(lags, xi_lag, edges)

    knyq = np.pi / xid.CELL
    kedges = np.logspace(np.log10(2 * np.pi / xid.LBOX), np.log10(knyq), 24)
    k_box, p_box = xid.measure_pk(delta_full, (xid.LBOX,) * 3, kedges)

    Ppad = tuple(n if p else 2 * n
                 for n, p in zip((Nx, Ny, Nz), xid.PERIODIC))
    pad = np.zeros(Ppad)
    pad[:Nx, :Ny, :Nz] = delta
    frac = (Nx * Ny * Nz) / float(np.prod(Ppad))
    k_pen, p_pen = xid.measure_pk(pad, tuple(n * xid.CELL for n in Ppad),
                                  kedges, mask_frac=frac)

    out = dict(seed=seed, r=r, xi=xi, nlag=nlag,
               k_box=k_box, p_box=p_box, k_pen=k_pen, p_pen=p_pen,
               ngrid=xid.NFULL, lbox=xid.LBOX, cell=xid.CELL,
               pencil_cells=np.array([Nx, Ny, Nz]),
               periodic=np.array(xid.PERIODIC), identity_maxrel=np.nan)

    if check_identity:
        xi_dir_lag = np.array([xid.s_direct(delta, di, dj, dk) / npair
                               for di, dj, dk, _, npair in lags])
        _, xi_dir, _ = xid.bin_xi(lags, xi_dir_lag, edges)
        rel = np.abs(xi_dir - xi) / np.maximum(np.abs(xi_dir), 1e-300)
        out["identity_maxrel"] = float(rel.max())
        out["xi_direct"] = xi_dir

    out["walltime"] = time.time() - t0
    return out


def write_txt(path, d):
    """Column format matching the HR-ICs-pipeline xi_*.txt outputs."""
    Nx, Ny, Nz = d["pencil_cells"]
    per = "".join(a for a, p in zip("xyz", d["periodic"]) if p) or "none"
    with open(path, "w") as f:
        f.write(f"# xi(r) from synthetic Gaussian field, seed={d['seed']}\n")
        f.write(f"# BoxSize={d['lbox']:g} Mpc/h  Ngrid={d['ngrid']}  "
                f"grid={Nx}x{Ny}x{Nz}  cell={d['cell']:.4g} Mpc/h\n")
        f.write(f"# Method=grid-fft-padded  bins=logarithmic  "
                f"periodic-axes={per}\n")
        f.write("# Columns: r[Mpc/h]  xi(r)  nlag\n")
        for r, x, n in zip(d["r"], d["xi"], d["nlag"]):
            f.write(f"{r:.6e}  {x:.8e}  {int(n):d}\n")


def cmd_compute(args):
    os.makedirs(args.outdir, exist_ok=True)
    lo, hi = args.seeds
    seeds = list(range(lo, hi + 1))
    print(f"pencil geometry from {os.path.normpath(_FIGDIR)}/plot_xi_identity.py: "
          f"{xid.NFULL}^3 box, L={xid.LBOX:g} Mpc/h, "
          f"pencil {xid.PENCIL}, periodic={xid.PERIODIC}")
    print(f"{len(seeds)} seed(s) requested -> {args.outdir}")

    done = skipped = 0
    for seed in seeds:
        path = os.path.join(args.outdir, f"seed_{seed:06d}.npz")
        if os.path.exists(path) and not args.force:
            skipped += 1
            continue
        d = measure_one(seed, args.rmin, args.rmax, args.nbins,
                        args.check_identity)
        np.savez(path, **d)
        if args.also_txt:
            write_txt(path.replace(".npz", ".txt"), d)
        msg = f"  seed {seed:6d}  {d['walltime']:6.2f}s  -> {os.path.basename(path)}"
        if args.check_identity:
            msg += f"   identity max rel.diff = {d['identity_maxrel']:.2e}"
        print(msg)
        done += 1

    print(f"\ncomputed {done}, skipped {skipped} already present "
          f"(use --force to recompute)")


# ------------------------------------------------------------------ plot ----
def cmd_plot(args):
    files = sorted(glob.glob(os.path.join(args.outdir, "seed_*.npz")))
    if not files:
        sys.exit(f"No seed_*.npz found in {args.outdir}")
    runs = [np.load(f) for f in files]
    print(f"loaded {len(runs)} run(s) from {args.outdir}")

    ident = np.array([float(d["identity_maxrel"]) for d in runs])
    if np.isfinite(ident).any():
        print(f"identity spot checks on {int(np.isfinite(ident).sum())} seed(s): "
              f"worst rel.diff = {np.nanmax(ident):.2e}")

    fig, (axp, axx) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # Individual realisations: thin and translucent so the spread reads as a
    # band; the median across seeds goes on top.
    thin = dict(lw=0.8, alpha=0.28, color="#1f77b4", zorder=2)

    for d in runs:
        axp.loglog(d["k_pen"], d["p_pen"], "-", **thin)
        axx.plot(d["r"], d["r"] ** 2 * d["xi"], "-", **thin)

    # A median needs a shared abscissa; all runs share it by construction,
    # but check rather than assume.
    r0, k0 = runs[0]["r"], runs[0]["k_pen"]
    same = all(len(d["r"]) == len(r0) and np.allclose(d["r"], r0) for d in runs)
    if same:
        xi_all = np.array([d["xi"] for d in runs])
        p_all = np.array([d["p_pen"] for d in runs])
        axx.plot(r0, r0 ** 2 * np.median(xi_all, axis=0), "-",
                 color="#d62728", lw=2.4, zorder=4,
                 label=f"median of {len(runs)} seeds")
        axp.loglog(k0, np.median(p_all, axis=0), "-",
                   color="#d62728", lw=2.4, zorder=4,
                   label=f"median of {len(runs)} seeds")
        lo, hi = np.percentile(xi_all, [16, 84], axis=0)
        axx.fill_between(r0, r0 ** 2 * lo, r0 ** 2 * hi, color="#d62728",
                         alpha=0.15, zorder=1, label="16–84 percentile")
    else:
        print("  r grids differ between runs; skipping median and percentiles")

    kt = np.logspace(np.log10(k0.min()), np.log10(k0.max()), 300)
    axp.loglog(kt, xid.power(kt), "--", color="0.35", lw=1.4, zorder=5,
               label="input linear theory")
    rt = np.logspace(np.log10(r0.min()), np.log10(r0.max()), 300)
    axx.plot(rt, rt ** 2 * xid.theory_xi(rt), "--", color="0.35", lw=1.4,
             zorder=5, label="input linear theory")

    axp.set_xlabel(r"$k$  [$h$/Mpc]")
    axp.set_ylabel(r"$P(k)$   $[(\mathrm{Mpc}/h)^3]$")
    axp.set_title("power spectrum (pencil, mask-convolved)", fontsize=10)
    axp.legend(frameon=False, fontsize=9, loc="lower left")

    axx.set_xscale("log")
    axx.axhline(0, color="k", lw=0.5, alpha=0.4)
    axx.set_xlabel(r"$r$  [Mpc/h]")
    axx.set_ylabel(r"$r^2\,\xi(r)$   $[(\mathrm{Mpc}/h)^2]$")
    axx.set_title("correlation function", fontsize=10)
    axx.legend(frameon=False, fontsize=9, loc="upper right")

    Nx, Ny, Nz = runs[0]["pencil_cells"]
    fig.suptitle(f"{len(runs)} seeds, pencil {Nx}$\\times${Ny}$\\times${Nz} "
                 f"cells in a ${runs[0]['ngrid']}^3$ box", fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


# ------------------------------------------------------------------ main ----
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="measure one .npz per seed")
    c.add_argument("--seeds", nargs=2, type=int, metavar=("LO", "HI"),
                   required=True, help="inclusive seed range")
    c.add_argument("--outdir", default="data/xi_sweep")
    c.add_argument("--rmin", type=float, default=10.0)
    c.add_argument("--rmax", type=float, default=200.0)
    c.add_argument("--nbins", type=int, default=28)
    c.add_argument("--check-identity", action="store_true",
                   help="also run the direct lag sum and record the worst "
                        "per-bin relative difference (slow)")
    c.add_argument("--also-txt", action="store_true",
                   help="additionally write xi_*.txt in column format")
    c.add_argument("--force", action="store_true",
                   help="recompute seeds whose .npz already exists")
    c.set_defaults(func=cmd_compute)

    q = sub.add_parser("plot", help="overlay every .npz in a directory")
    q.add_argument("--outdir", default="data/xi_sweep")
    q.add_argument("--output", default="plots/xi_sweep.png")
    q.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
