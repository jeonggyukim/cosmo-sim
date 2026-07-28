#!/usr/bin/env python3
"""
xi_identity.py — show that the two grid ksi(r) estimators in
tools/pk_pipeline are the same estimator, evaluated two different ways.

  method A  direct lag sum      s(d) = sum_x delta(x) delta(x+d)      (measure_xi_mpi.c)
  method B  zero-padded FFT     s    = IFFT(|FFT(delta)|^2)           (measure_xi_fft_mpi.c)

Both divide by the same analytic pair count n(d), which for a rectangular
region factorises per axis:

    periodic axis     n = N          (circular autocorrelation, full count)
    non-periodic axis n = N - |d|    (linear autocorrelation, overlap count)

That n(d) is the mask autocorrelation sum_x W(x)W(x+d), i.e. the RR term of
Landy-Szalay written in closed form, which is why no random catalogue is needed.

Method B is not an approximation of method A. By DFT orthogonality,

    IFFT(|FFT(delta)|^2)(d) = sum_x sum_y delta_x delta_y [x-y = d mod N]
                            = sum_y delta_y delta_{y+d}

which is method A exactly. The two differ only in the order the same finite set
of products is added up, so they agree to floating-point roundoff.

Geometry here mimics the pencil beam used in the seed_35211 run: a column cut
out of a periodic box, narrow in x and y, spanning the full box in z. The two
cut axes are non-periodic (real edges, zero-padded to 2N); the long axis stays
periodic (not padded, wrap-around is a physically real pair).

Figure for notes/xi_estimators.tex.

Run:
    make -C notes figures
or, from this directory:
    conda run -n cosmo python plot_xi_identity.py

Writes xi_identity.pdf and xi_identity.png next to this script.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- knobs ----
NFULL       = 128                  # full-box grid
LBOX        = 1000.0               # Mpc/h
PENCIL      = (0.4375, 0.5625)     # x,y sub-selection -> 16 cells = 125 Mpc/h
PERIODIC    = (False, False, True) # x,y cut; z spans the box
NBINS       = 28
RMIN, RMAX  = 10.0, 200.0          # Mpc/h
SEED        = 35211

OMEGA_M, HUBBLE, NS, SIGMA8 = 0.3085, 0.6774, 0.9682, 0.8228

CELL = LBOX / NFULL
VBOX = LBOX ** 3


# ------------------------------------------------------- input spectrum ----
def bbks_pk(k):
    """BBKS transfer function, arbitrary amplitude. k in h/Mpc."""
    k = np.asarray(k, dtype=float)
    q = np.where(k > 0, k / (OMEGA_M * HUBBLE), 1e-30)
    t = (np.log(1 + 2.34 * q) / (2.34 * q)) * (
        1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** -0.25
    return np.where(k > 0, k ** NS * t ** 2, 0.0)


def _sigma8_sq(norm):
    k = np.logspace(-4, 2, 4000)
    x = k * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.trapezoid(norm * bbks_pk(k) * k ** 2 * w ** 2, k) / (2 * np.pi ** 2)


PK_NORM = SIGMA8 ** 2 / _sigma8_sq(1.0)
power = lambda k: PK_NORM * bbks_pk(k)


def theory_xi(r):
    """ksi(r) = 1/(2 pi^2) int P(k) k^2 j0(kr) dk."""
    k = np.logspace(-4, 1.4, 6000)
    pk = power(k)
    out = np.empty(len(np.atleast_1d(r)))
    for i, rr in enumerate(np.atleast_1d(r)):
        j0 = np.sinc(k * rr / np.pi)
        out[i] = np.trapezoid(pk * k ** 2 * j0, k) / (2 * np.pi ** 2)
    return out


# ------------------------------------------- Gaussian field on full box ----
def make_field(seed):
    rng = np.random.default_rng(seed)
    kf = 2 * np.pi * np.fft.fftfreq(NFULL, d=CELL)
    kz = 2 * np.pi * np.fft.rfftfreq(NFULL, d=CELL)
    kk = np.sqrt(kf[:, None, None] ** 2 + kf[None, :, None] ** 2 + kz[None, None, :] ** 2)

    wk = np.fft.rfftn(rng.standard_normal((NFULL, NFULL, NFULL)))
    dk = wk * np.sqrt(power(kk) * NFULL ** 3 / VBOX)
    dk[0, 0, 0] = 0.0
    return np.fft.irfftn(dk, s=(NFULL, NFULL, NFULL), axes=(0, 1, 2))


# ---------------------------------------------------- the two estimators ----
def lag_list(shape, cells, rmin, rmax):
    """The lag loop the C code walks: half-space, capped by k_max and by the
    interior size, skipping lags with no overlapping pairs."""
    Nx, Ny, Nz = shape
    kmax = min(int(np.ceil(rmax / min(cells))), max(shape) - 1)
    out = []
    for di in range(kmax + 1):
        if PERIODIC[0] and di > Nx // 2:
            break
        for dj in range(kmax + 1):
            if PERIODIC[1] and dj > Ny // 2:
                break
            for dk in range(kmax + 1):
                if PERIODIC[2] and dk > Nz // 2:
                    break
                if di == dj == dk == 0:
                    continue
                nx = Nx if PERIODIC[0] else Nx - di
                ny = Ny if PERIODIC[1] else Ny - dj
                nz = Nz if PERIODIC[2] else Nz - dk
                if min(nx, ny, nz) <= 0:
                    continue
                r = np.sqrt((di * cells[0]) ** 2 + (dj * cells[1]) ** 2
                            + (dk * cells[2]) ** 2)
                if rmin <= r <= rmax:
                    out.append((di, dj, dk, r, nx * ny * nz))
    return out


def s_direct(d, di, dj, dk):
    """sum_x delta(x) delta(x+lag), wrapping only on periodic axes."""
    Nx, Ny, Nz = d.shape
    ix = np.arange(Nx if PERIODIC[0] else Nx - di)
    iy = np.arange(Ny if PERIODIC[1] else Ny - dj)
    iz = np.arange(Nz if PERIODIC[2] else Nz - dk)
    a = d[np.ix_(ix, iy, iz)]
    b = d[np.ix_((ix + di) % Nx, (iy + dj) % Ny, (iz + dk) % Nz)]
    return float(np.einsum("ijk,ijk->", a, b))


def s_padded_fft(d):
    """IFFT(|FFT(delta)|^2) on the per-axis padded grid: all lags at once."""
    Nx, Ny, Nz = d.shape
    P = tuple(n if p else 2 * n for n, p in zip((Nx, Ny, Nz), PERIODIC))
    pad = np.zeros(P)
    pad[:Nx, :Ny, :Nz] = d
    fk = np.fft.rfftn(pad, s=P, axes=(0, 1, 2))
    return np.fft.irfftn(np.abs(fk) ** 2, s=P, axes=(0, 1, 2))


def measure_pk(field, box, kedges, mask_frac=1.0):
    """Binned P(k) of a gridded field.

    For a field on M cells spanning volume V, the continuum transform is
    approximated by (V/M) * FFT, so P(k) = |FFT|^2 * V / M^2. When the field
    has been zero-padded, mask_frac = V_occupied/V is divided out so the
    large-scale amplitude is comparable to the unpadded case.
    """
    n = field.shape
    M = field.size
    V = float(np.prod(box))
    fk = np.fft.rfftn(field, s=n, axes=(0, 1, 2))
    pk = np.abs(fk) ** 2 * V / M ** 2 / mask_frac

    kx = 2 * np.pi * np.fft.fftfreq(n[0], d=box[0] / n[0])
    ky = 2 * np.pi * np.fft.fftfreq(n[1], d=box[1] / n[1])
    kz = 2 * np.pi * np.fft.rfftfreq(n[2], d=box[2] / n[2])
    kk = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                 + kz[None, None, :] ** 2)

    # rfftn stores only half the modes: every kz plane except the first and
    # (for even n) the Nyquist plane stands for two Hermitian partners.
    w = np.full(kk.shape, 2.0)
    w[:, :, 0] = 1.0
    if n[2] % 2 == 0:
        w[:, :, -1] = 1.0

    kk, pk, w = kk.ravel(), pk.ravel(), w.ravel()
    good = kk > 0
    idx = np.digitize(kk[good], kedges) - 1
    nb = len(kedges) - 1
    wsum = np.bincount(idx, weights=w[good], minlength=nb)[:nb]
    psum = np.bincount(idx, weights=w[good] * pk[good], minlength=nb)[:nb]
    ksum = np.bincount(idx, weights=w[good] * kk[good], minlength=nb)[:nb]
    ok = wsum > 0
    return ksum[ok] / wsum[ok], psum[ok] / wsum[ok]


def bin_xi(lags, xi_of_lag, edges):
    ns = np.zeros(len(edges) - 1)
    xs = np.zeros(len(edges) - 1)
    rs = np.zeros(len(edges) - 1)
    for (_, _, _, r, _), xi in zip(lags, xi_of_lag):
        b = np.searchsorted(edges, r, side="right") - 1
        if 0 <= b < len(ns):
            ns[b] += 1
            xs[b] += xi
            rs[b] += r
    ok = ns > 0
    return rs[ok] / ns[ok], xs[ok] / ns[ok], ns[ok]


# ------------------------------------------------------------------ main ----
def main():
    print(f"full box {NFULL}^3, L={LBOX:g} Mpc/h, cell={CELL:.4g} Mpc/h")
    delta_full = make_field(SEED)

    i0, i1 = int(PENCIL[0] * NFULL), int(PENCIL[1] * NFULL)
    delta = delta_full[i0:i1, i0:i1, :].copy()
    delta -= delta.mean()                       # interior-mean normalisation
    Nx, Ny, Nz = delta.shape
    cells = (CELL, CELL, CELL)
    print(f"pencil interior {Nx}x{Ny}x{Nz} cells "
          f"= {Nx*CELL:.4g} x {Ny*CELL:.4g} x {Nz*CELL:.4g} Mpc/h")
    print("periodic axes: "
          + (''.join(a for a, p in zip('xyz', PERIODIC) if p) or '(none)'))

    lags = lag_list((Nx, Ny, Nz), cells, RMIN, RMAX)
    print(f"{len(lags)} lag triples in r=[{RMIN:g},{RMAX:g}] Mpc/h")

    num = s_padded_fft(delta)
    print("padded grid {}x{}x{}".format(*num.shape))

    xi_dir = np.array([s_direct(delta, di, dj, dk) / npair
                       for di, dj, dk, _, npair in lags])
    xi_fft = np.array([num[di, dj, dk] / npair for di, dj, dk, _, npair in lags])

    rel = np.abs(xi_dir - xi_fft) / np.maximum(np.abs(xi_dir), 1e-300)
    print(f"\nper-lag agreement:  worst rel.diff = {rel.max():.3e}   "
          f"median = {np.median(rel):.3e}   (double eps = {np.finfo(float).eps:.3e})")

    edges = np.logspace(np.log10(RMIN), np.log10(RMAX), NBINS + 1)
    r_d, x_d, n_d = bin_xi(lags, xi_dir, edges)
    r_f, x_f, _ = bin_xi(lags, xi_fft, edges)
    rel_bin = np.abs(x_d - x_f) / np.maximum(np.abs(x_d), 1e-300)
    print(f"per-bin agreement:  worst rel.diff = {rel_bin.max():.3e}\n")

    print(f"{'r [Mpc/h]':>10} {'xi direct':>18} {'xi padded-FFT':>18} "
          f"{'rel.diff':>10} {'nlag':>6}")
    for r, a, b, e, n in zip(r_d, x_d, x_f, rel_bin, n_d):
        print(f"{r:10.3f} {a:18.10e} {b:18.10e} {e:10.2e} {int(n):6d}")

    # ---------------------------------------------------------- P(k) ----
    # |FFT(delta_pad)|^2 is the intermediate the padded-FFT estimator forms
    # on its way to xi: same array, before the inverse transform. Shown next
    # to the full-box P(k) so the effect of the mask is visible.
    knyq = np.pi / CELL
    kedges = np.logspace(np.log10(2 * np.pi / LBOX), np.log10(knyq), 24)

    k_box, p_box = measure_pk(delta_full, (LBOX,) * 3, kedges)

    Ppad = tuple(n if p else 2 * n for n, p in zip((Nx, Ny, Nz), PERIODIC))
    pad = np.zeros(Ppad)
    pad[:Nx, :Ny, :Nz] = delta
    box_pad = tuple(n * CELL for n in Ppad)
    frac = (Nx * Ny * Nz) / float(np.prod(Ppad))
    k_pen, p_pen = measure_pk(pad, box_pad, kedges, mask_frac=frac)

    # ------------------------------------------------------------- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2),
                             gridspec_kw=dict(height_ratios=[3, 1],
                                              hspace=0.08, wspace=0.24))
    (axp, axx), (axpr, axxr) = axes

    # ---- left: power spectrum ----
    kt = np.logspace(np.log10(kedges[0]), np.log10(kedges[-1]), 300)
    axp.loglog(kt, power(kt), color="0.6", lw=1.2, ls="--", zorder=1,
               label="input linear theory")
    axp.loglog(k_box, p_box, "s", ms=5, color="#1f77b4", zorder=2,
               label=f"full box ${NFULL}^3$, all axes periodic")
    axp.loglog(k_pen, p_pen, "o", ms=5.5, mfc="none", mew=1.6,
               color="#d62728", zorder=3,
               label=r"pencil, $|\mathrm{FFT}(\delta_\mathrm{pad})|^2$")
    axp.set_ylabel(r"$P(k)$   $[(\mathrm{Mpc}/h)^3]$")
    axp.legend(frameon=False, fontsize=9, loc="lower left")
    axp.set_title("power spectrum", fontsize=10)

    axpr.semilogx(k_box, p_box / power(k_box), "s-", ms=4, color="#1f77b4",
                  label="full box")
    axpr.semilogx(k_pen, p_pen / power(k_pen), "o-", ms=4, mfc="none",
                  color="#d62728", label="pencil (mask-convolved)")
    axpr.axhline(1.0, color="k", lw=0.8, ls=":")
    axpr.set_ylim(0, 2.2)
    axpr.set_xlabel(r"$k$  [$h$/Mpc]")
    axpr.set_ylabel(r"$P / P_\mathrm{theory}$", fontsize=9)
    axpr.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axpr.grid(alpha=0.25, which="both", lw=0.4)

    # ---- right: correlation function, the two estimators ----
    rt = np.logspace(np.log10(RMIN), np.log10(RMAX), 200)
    axx.plot(rt, rt ** 2 * theory_xi(rt), color="0.6", lw=1.2, ls="--",
             zorder=1, label="input linear theory")
    axx.plot(r_d, r_d ** 2 * x_d, "-", color="#1f77b4", lw=2.4, zorder=2,
             label="direct lag sum  (measure_xi_mpi)")
    axx.plot(r_f, r_f ** 2 * x_f, "o", ms=5.5, mfc="none", mew=1.6,
             color="#d62728", zorder=3,
             label="padded FFT  (measure_xi_fft_mpi)")
    axx.set_xscale("log")
    axx.set_ylabel(r"$r^2\,\xi(r)$   $[(\mathrm{Mpc}/h)^2]$")
    axx.axhline(0, color="k", lw=0.5, alpha=0.4)
    axx.legend(frameon=False, fontsize=9, loc="upper right")
    axx.set_title("correlation function", fontsize=10)

    axxr.semilogy(r_d, np.maximum(rel_bin, 1e-18), "s-", ms=4, color="#2ca02c")
    axxr.axhline(np.finfo(float).eps, color="k", ls=":", lw=1,
                 label=r"double $\epsilon = 2.2\times10^{-16}$")
    axxr.set_ylim(1e-18, 1e-10)
    axxr.set_xscale("log")
    axxr.set_xlabel(r"$r$  [Mpc/h]")
    axxr.set_ylabel(r"$|\Delta\xi| / |\xi|$", fontsize=9)
    axxr.legend(frameon=False, fontsize=8, loc="upper left")
    axxr.grid(alpha=0.25, which="both", lw=0.4)

    fig.suptitle(f"pencil {Nx}$\\times${Ny}$\\times${Nz} cells "
                 f"({Nx*CELL:.0f}$\\times${Ny*CELL:.0f}$\\times${Nz*CELL:.0f} "
                 "Mpc/h), x,y non-periodic (padded to 2N), z periodic",
                 fontsize=11, y=0.97)

    fig.savefig("xi_identity.pdf", bbox_inches="tight")
    fig.savefig("xi_identity.png", dpi=150, bbox_inches="tight")
    print("\nSaved: xi_identity.pdf, xi_identity.png")
    print("Left: the pencil P(k) is the true P(k) convolved with the mask "
          "window, so it is not expected to track theory; the full box is.")
    print("Right: theory is context only. One pencil is cosmic-variance "
          "limited and the grid applies a cell window. The two estimators "
          "lying on top of each other is the point.")


if __name__ == "__main__":
    main()
