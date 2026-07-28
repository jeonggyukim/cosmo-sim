#!/usr/bin/env python3
"""
xi_toy.py — the smallest honest demonstration that the direct lag sum and the
zero-padded FFT are the same estimator.

Deliberately tiny and 2D so that every number can be checked by hand.

Part 1 is a four-cell 1D array where the wrap-around term can be written out
in full, showing what padding removes and why "pad to 2N" is exact rather
than a safety margin.

Part 2 is a small 2D grid with one cut axis and one periodic axis, mimicking
a pencil beam in cross-section. Both estimators are evaluated for every lag
and printed side by side, plus a figure of the two lag planes and their
difference.

Both parts keep the direct lag sum. That is the point of a toy: it is cheap
enough that you never have to take the fast path on trust. For sweeps over
many seeds, where the cost of the direct sum stops being negligible, use
scripts/analysis/xi_sweep.py, which runs the FFT path only.

Figure for notes/xi_estimators.tex.

Run:
    make -C notes figures
or, from this directory:
    conda run -n cosmo python plot_xi_toy.py

Writes xi_toy.pdf and xi_toy.png next to this script.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Part 1 — one dimension, four cells, done by hand
# ===========================================================================
def part1():
    print("=" * 74)
    print("PART 1  wrap-around in 1D, N=4, lag=1")
    print("=" * 74)

    a, b, c, d = 2.0, -1.0, 3.0, -4.0
    f = np.array([a, b, c, d])
    print(f"  f = (a,b,c,d) = ({a:g}, {b:g}, {c:g}, {d:g})\n")

    # By hand.
    lin = a * b + b * c + c * d
    wrap = d * a
    print(f"  linear   s(1) = ab + bc + cd            = {lin:g}")
    print(f"  wrap term       da                      = {wrap:g}")
    print(f"  circular s(1) = ab + bc + cd + da       = {lin + wrap:g}\n")

    # Unpadded FFT: index arithmetic is modulo 4, so it returns the circular sum.
    circ_fft = np.fft.irfft(np.abs(np.fft.rfft(f)) ** 2, n=4)[1]
    print(f"  unpadded FFT, IFFT(|FFT(f)|^2)[1]       = {circ_fft:.15g}")
    print(f"    matches circular?  {np.isclose(circ_fft, lin + wrap)}")

    # Padded to P=8: every wrap term now contains a factor from the zero zone.
    fp = np.concatenate([f, np.zeros(4)])
    lin_fft = np.fft.irfft(np.abs(np.fft.rfft(fp)) ** 2, n=8)[1]
    print(f"  padded to P=8, IFFT(|FFT(f_pad)|^2)[1]  = {lin_fft:.15g}")
    print(f"    matches linear?    {np.isclose(lin_fft, lin)}\n")

    # The padding condition d <= P - N, checked at every lag.
    print("  padding condition  d <= P - N,  here P-N = 4:")
    print(f"  {'lag':>5} {'direct linear':>15} {'padded FFT':>15} {'exact?':>8}")
    num = np.fft.irfft(np.abs(np.fft.rfft(fp)) ** 2, n=8)
    for lag in range(5):
        direct = float(sum(f[x] * f[x + lag] for x in range(4 - lag))) if lag < 4 else 0.0
        print(f"  {lag:5d} {direct:15.10g} {num[lag]:15.10g} "
              f"{str(np.isclose(direct, num[lag])):>8}")
    print()


# ===========================================================================
# Part 2 — two dimensions, one cut axis and one periodic axis
# ===========================================================================
NX, NY = 8, 16
PERIODIC = (False, True)      # x is cut (real edges); y spans a periodic box
SEED = 35211
DMAX = 5                      # print lags 0..DMAX on each axis


def s_direct(f, dx, dy):
    """sum_x f(x) f(x+lag), wrapping only on the periodic axis."""
    nx, ny = f.shape
    ix = np.arange(nx if PERIODIC[0] else nx - dx)
    iy = np.arange(ny if PERIODIC[1] else ny - dy)
    a = f[np.ix_(ix, iy)]
    b = f[np.ix_((ix + dx) % nx, (iy + dy) % ny)]
    return float((a * b).sum())


def s_padded_fft(f):
    """IFFT(|FFT(f)|^2) on the per-axis padded grid: every lag at once."""
    nx, ny = f.shape
    P = tuple(n if p else 2 * n for n, p in zip((nx, ny), PERIODIC))
    pad = np.zeros(P)
    pad[:nx, :ny] = f
    return np.fft.irfftn(np.abs(np.fft.rfftn(pad, s=P, axes=(0, 1))) ** 2,
                         s=P, axes=(0, 1))


def npair(dx, dy):
    """Mask autocorrelation, separable because the region is a rectangle."""
    nx = NX if PERIODIC[0] else NX - dx
    ny = NY if PERIODIC[1] else NY - dy
    return nx * ny


def part2():
    print("=" * 74)
    print(f"PART 2  {NX}x{NY} grid, x non-periodic (padded to {2*NX}), "
          f"y periodic (not padded)")
    print("=" * 74)

    rng = np.random.default_rng(SEED)
    f = rng.standard_normal((NX, NY))
    f -= f.mean()                       # interior-mean normalisation
    num = s_padded_fft(f)
    print(f"  padded grid {num.shape[0]}x{num.shape[1]}\n")

    print(f"  {'lag':>9} {'n(d)':>6} {'xi direct':>20} {'xi padded FFT':>20} "
          f"{'rel.diff':>11}")
    worst = 0.0
    xi_dir = np.full((DMAX + 1, DMAX + 1), np.nan)
    xi_fft = np.full((DMAX + 1, DMAX + 1), np.nan)
    for dx in range(DMAX + 1):
        for dy in range(DMAX + 1):
            if dx == dy == 0:
                continue
            n = npair(dx, dy)
            xd = s_direct(f, dx, dy) / n
            xf = num[dx, dy] / n
            xi_dir[dx, dy], xi_fft[dx, dy] = xd, xf
            rel = abs(xd - xf) / max(abs(xd), 1e-300)
            worst = max(worst, rel)
            print(f"  ({dx},{dy}){'':>3} {n:6d} {xd:20.14e} {xf:20.14e} {rel:11.2e}")

    print(f"\n  worst relative difference: {worst:.3e}   "
          f"(double eps = {np.finfo(float).eps:.3e})")

    # One entry spelled out, so the table is checkable rather than merely printed.
    dx, dy = 1, 0
    terms = [f[x, y] * f[x + dx, y] for x in range(NX - dx) for y in range(NY)]
    print(f"\n  hand check of lag (1,0): {len(terms)} products, "
          f"n(d) = {npair(1,0)}")
    print(f"    sum of products / n(d) = {sum(terms) / npair(1,0):.14e}")
    print(f"    table value            = {xi_dir[1,0]:.14e}")

    # ------------------------------------------------------------- plot ----
    diff = np.abs(xi_dir - xi_fft)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    vmax = np.nanmax(np.abs(xi_dir))

    for ax, img, title in zip(
            axes[:2], (xi_dir, xi_fft),
            ("direct lag sum", "padded FFT")):
        im = ax.imshow(img.T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"lag $d_x$  (cut axis)")
        ax.set_ylabel(r"lag $d_y$  (periodic axis)")
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\xi$(lag)")

    im = axes[2].imshow(np.log10(np.maximum(diff, 1e-20)).T, origin="lower",
                        cmap="viridis", vmin=-20, vmax=-14)
    axes[2].set_title(r"$\log_{10}|\Delta\xi|$", fontsize=10)
    axes[2].set_xlabel(r"lag $d_x$")
    axes[2].set_ylabel(r"lag $d_y$")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle(f"{NX}$\\times${NY} toy: identical lag planes, "
                 f"difference at the {worst:.0e} level", fontsize=11)
    fig.tight_layout()
    fig.savefig("xi_toy.pdf", bbox_inches="tight")
    fig.savefig("xi_toy.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: xi_toy.pdf, xi_toy.png")


if __name__ == "__main__":
    part1()
    part2()
