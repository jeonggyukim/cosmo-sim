#!/usr/bin/env python3
"""
plot_seed_conditioning.py — what a random-seed search does to a simulation.

Box of L = 1000 Mpc/h on a 256^3 grid; subvolume 125 x 125 x 1000 Mpc/h (1/8 of
the box side in x and y, the full extent in z), holding 1/64 of the volume.
Gaussian fields with the CLASS linear power spectrum renormalized to
sigma_8 = 0.8. Only linear fields are needed, since the modes responsible for
the effect grow linearly.

600 realizations are generated. In each one xi(r) is measured twice: over the
full box, and over the subvolume with the mean density taken from the subvolume
itself, so the measurement carries the same integral constraint a real analysis
carries. The ensemble mean of the full-box measurement is used as "theory",
which removes gridding and binning differences from the comparison.

A seed search is then imitated: for each seed, the squared residuals against
theory over r = 60-130 Mpc/h, each normalized by the scatter of its bin, are
summed to give a search score; the 20% of seeds with the lowest score are kept.

The measurement takes several minutes, so it is cached in
seed_conditioning_cache.npz and reused whenever the parameters still match.

Figure for notes/seed_selection.tex. Four panels:
  (a) all realizations with their 16-84 band, no selection
  (b) the seed of 600 with the lowest search score
  (c) scatter retained within the matched range vs the accepted fraction,
      with the corresponding point for rejection on subvolume density alone
  (d) one separation outside the matched range, before and after selection

Run:
    make -C notes figures
or, from this directory:
    conda run -n cosmo python plot_seed_conditioning.py

Writes seed_conditioning.pdf and seed_conditioning.png next to this script,
and prints the numbers quoted in the note.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- knobs ----
NGRID  = 256
LBOX   = 1000.0             # Mpc/h
NZOOM  = 32                 # 32 cells x 3.90625 Mpc/h = 125 Mpc/h
NREAL  = 600
KEEP   = 0.20               # fraction of seeds the imitated search keeps
FITLO, FITHI = 60.0, 130.0  # Mpc/h, the range the search matches
RBIN   = 5.0                # Mpc/h
RMAX   = 160.0              # Mpc/h
SEED   = 12345
SIGMA8 = 0.8
BROKEN = 0.9                # the broken code reports xi 10% low
NLOW   = 8                  # k_z = n 2pi/L, n = 1..8, i.e. k <= 0.05 h/Mpc

PKFILE = "../../../data/class_pk_z200_pk.dat"
AXES = (0, 1, 2)
CELL = LBOX / NGRID
NCELL = NGRID ** 3


def power_spectrum():
    """CLASS linear P(k), renormalized so that sigma_8 = SIGMA8."""
    table = np.loadtxt(PKFILE, comments="#")
    lk, lp = np.log(table[:, 0]), np.log(table[:, 1])

    def p_of_k(k):
        k = np.clip(k, table[0, 0], table[-1, 0])
        return np.exp(np.interp(np.log(k), lk, lp))

    k = np.logspace(np.log10(table[0, 0]), np.log10(table[-1, 0]), 4000)
    x = k * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    s2 = np.trapezoid(k ** 2 * p_of_k(k) * w ** 2, k) / (2 * np.pi ** 2)
    return lambda k: (SIGMA8 ** 2 / s2) * p_of_k(k)


def measure():
    rng = np.random.default_rng(SEED)
    p_of_k = power_spectrum()

    kx = 2 * np.pi * np.fft.fftfreq(NGRID, d=CELL)
    kz = 2 * np.pi * np.fft.rfftfreq(NGRID, d=CELL)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    pk = p_of_k(kmag)
    pk[0, 0, 0] = 0.0
    amp = np.sqrt(pk * NCELL / LBOX ** 3)
    del KX, KY, KZ, kmag, pk

    sx = np.fft.fftfreq(NGRID) * LBOX
    SX, SY, SZ = np.meshgrid(sx, sx, sx, indexing="ij")
    sep = np.sqrt(SX ** 2 + SY ** 2 + SZ ** 2).ravel()
    del SX, SY, SZ
    edges = np.arange(0.0, RMAX, RBIN)
    nbin = len(edges) - 1
    which = np.digitize(sep, edges)
    del sep
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = np.zeros((NGRID, NGRID, NGRID))
    mask[:NZOOM, :NZOOM, :] = 1.0
    fm = np.fft.rfftn(mask)
    pairs = np.fft.irfftn(np.abs(fm) ** 2, s=(NGRID,) * 3, axes=AXES).ravel()
    den_zoom = np.bincount(which, weights=pairs, minlength=nbin + 2)[1:nbin + 1]
    den_box = np.bincount(which, minlength=nbin + 2)[1:nbin + 1] * float(NCELL)
    del fm, pairs

    xi_box = np.empty((NREAL, nbin))
    xi_zoom = np.empty((NREAL, nbin))
    p_low = np.empty(NREAL)
    dbar = np.empty(NREAL)

    for i in range(NREAL):
        dk = np.fft.rfftn(rng.standard_normal((NGRID,) * 3)) * amp
        delta = np.fft.irfftn(dk, s=(NGRID,) * 3, axes=AXES)

        c = np.fft.irfftn(np.abs(dk) ** 2, s=(NGRID,) * 3, axes=AXES).ravel()
        xi_box[i] = np.bincount(which, weights=c,
                                minlength=nbin + 2)[1:nbin + 1] / den_box

        local = delta[:NZOOM, :NZOOM, :]
        dbar[i] = local.mean()
        cut = (delta - local.mean()) * mask
        z = np.fft.rfftn(cut)
        c = np.fft.irfftn(np.abs(z) ** 2, s=(NGRID,) * 3, axes=AXES).ravel()
        xi_zoom[i] = np.bincount(which, weights=c,
                                 minlength=nbin + 2)[1:nbin + 1] / den_zoom

        column = local.mean(axis=(0, 1))
        p_low[i] = np.sum(np.abs(np.fft.rfft(column - column.mean())
                                 [1:NLOW + 1]) ** 2) / NGRID ** 2

        if (i + 1) % 100 == 0:
            print(f"realization {i + 1}/{NREAL}", flush=True)

    return centers, xi_box, xi_zoom, p_low, dbar


CACHE = "seed_conditioning_cache.npz"


def load_or_measure():
    """The 600 realizations take several minutes at NGRID = 256, so the
    measurement is cached and reused whenever the parameters still match."""
    key = np.array([NGRID, LBOX, NZOOM, NREAL, RBIN, RMAX, SEED, SIGMA8])
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if d["key"].shape == key.shape and np.all(d["key"] == key):
            print(f"reusing {CACHE}")
            return (d["r"], d["xi_box"], d["xi_zoom"], d["p_low"], d["dbar"])
        print(f"{CACHE} was made with other parameters; measuring again")
    out = measure()
    np.savez(CACHE, key=key, r=out[0], xi_box=out[1], xi_zoom=out[2],
             p_low=out[3], dbar=out[4])
    return out


def main():
    r, xi_box, xi_zoom, p_low, dbar = load_or_measure()

    theory = xi_box.mean(axis=0)
    fair_mean = xi_zoom.mean(axis=0)
    fair_std = xi_zoom.std(axis=0)

    fit = np.where((r > FITLO) & (r < FITHI))[0]
    chi2 = (((xi_zoom[:, fit] - theory[fit]) / fair_std[fit]) ** 2).sum(axis=1)
    kept = chi2 <= np.quantile(chi2, KEEP)
    best = int(np.argmin(chi2))

    sel_mean = xi_zoom[kept].mean(axis=0)
    sel_std = xi_zoom[kept].std(axis=0)
    ratio = sel_std / fair_std

    print("\n--- numbers quoted in notes/seed_selection.tex ---")
    print(f"realizations {NREAL}, kept {kept.sum()}, "
          f"matched range {FITLO:.0f}-{FITHI:.0f} Mpc/h ({len(fit)} bins)")

    j95 = int(np.argmin(np.abs(r - 95.0)))
    print(f"\nr = {r[j95]:.0f}: spread(region) = {fair_std[j95]:.3e}, "
          f"spread(full box) = {xi_box[:, j95].std():.3e}, "
          f"ratio = {fair_std[j95] / xi_box[:, j95].std():.1f}")
    print(f"r = {r[j95]:.0f}: average measured minus theory "
          f"= {fair_mean[j95] - theory[j95]:+.2e} "
          f"= {(fair_mean[j95] - theory[j95]) / fair_std[j95]:+.2f} spreads")

    print("\nper separation, all seeds -> kept seeds:")
    print(f"{'r':>6} {'matched':>8} {'spread ratio':>13} {'shift/spread':>13}"
          f" {'shift/s.e.':>11} {'spread/|theory|':>16}")
    for j in range(len(r)):
        if r[j] < 25 or r[j] > 155:
            continue
        shift = (sel_mean[j] - fair_mean[j]) / fair_std[j]
        se = fair_std[j] / np.sqrt(kept.sum())
        print(f"{r[j]:6.0f} {str(j in fit):>8} {ratio[j]:13.2f} "
              f"{shift:13.2f} {(sel_mean[j] - fair_mean[j]) / se:11.1f} "
              f"{fair_std[j] / abs(theory[j]):16.2f}")

    lo = p_low[kept]
    print(f"\nlargest-scale power of the region (k <= 0.05 h/Mpc): all seeds "
          f"{p_low.mean():.2e} +- {p_low.std():.2e}, kept seeds spread ratio "
          f"{lo.std() / p_low.std():.2f}, average moved "
          f"{(lo.mean() - p_low.mean()) / p_low.std():+.2f} spreads")
    print(f"correlation of that power with the search score: "
          f"{np.corrcoef(chi2, p_low)[0, 1]:+.2f}")

    dev = (xi_zoom[np.ix_(kept, fit)] - theory[fit]) / fair_std[fit]
    dev_best = (xi_zoom[best, fit] - theory[fit]) / fair_std[fit]
    j65 = int(np.argmin(np.abs(r - 65.0)))
    print(f"\nkept seeds still differ from theory by "
          f"{float(np.sqrt((dev ** 2).mean())):.2f} spreads per separation")
    print(f"the single best of {NREAL} seeds differs by "
          f"{float(np.sqrt((dev_best ** 2).mean())):.2f} spreads per "
          "separation")
    print(f"a code reporting xi {100 * (1 - BROKEN):.0f}% low moves "
          f"r = {r[j65]:.0f} by "
          f"{(1 - BROKEN) * abs(theory[j65]) / fair_std[j65]:.3f} spreads, "
          "far inside what the search accepts")

    fracs = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    damage = []
    for f in fracs:
        sub = chi2 <= np.quantile(chi2, f)
        damage.append(float(np.mean(xi_zoom[sub][:, fit].std(axis=0)
                                    / fair_std[fit])))
    damage = np.array(damage)
    print("\nfraction of seeds kept -> spread over the matched range:")
    for f, d in zip(fracs, damage):
        print(f"  {f:5.0%} ({int(round(f * NREAL)):3d} seeds)   {d:.2f}")

    def cut_cost(mask_, name):
        sub = xi_zoom[mask_][:, fit]
        keeps = float(mask_.mean())
        spread = float(np.mean(sub.std(axis=0) / fair_std[fit]))
        moves = float(np.mean((sub.mean(axis=0) - fair_mean[fit])
                              / fair_std[fit]))
        print(f"  {name}: keeps {keeps:5.1%}, spread over the matched range "
              f"{spread:.2f}, average moves {moves:+.2f}")
        return keeps, spread

    print("\ntwo milder rules, for comparison:")
    within = np.all(np.abs(xi_zoom[:, fit] - theory[fit])
                    < 2 * fair_std[fit], axis=1)
    xi_cut = cut_cost(within, "reject on xi, beyond 2 spreads anywhere    ")
    dens_cut = cut_cost(np.abs(dbar - dbar.mean()) < 2 * dbar.std(),
                        "reject on the density of the region only  ")

    # ------------------------------------------------------------ figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.4))
    w = r ** 2
    show = r <= 155

    a = ax[0, 0]
    lo16, hi84 = np.percentile(xi_zoom, [16, 84], axis=0)
    for row in xi_zoom[:80]:
        a.plot(r[show], (w * row)[show], color="0.75", lw=0.4, alpha=0.6,
               zorder=1)
    a.fill_between(r[show], (w * lo16)[show], (w * hi84)[show], color="C0",
                   alpha=0.28, zorder=2, label="middle 68% of the seeds")
    a.plot(r[show], (w * fair_mean)[show], color="C0", lw=2.0, zorder=3,
           label="average over seeds")
    a.plot(r[show], (w * theory)[show], color="k", lw=2.0, ls="--", zorder=4,
           label="theory")
    a.axhline(0.0, color="0.5", lw=0.5)
    a.set_xlabel(r"$r$ [Mpc/$h$]")
    a.set_ylabel(r"$r^2\,\xi(r)$ [(Mpc/$h$)$^2$]")
    a.set_title("(a) a subvolume this size scatters this much", fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="lower left")

    a = ax[0, 1]
    near = r <= FITHI + 5.0
    a.axvspan(FITLO, FITHI, color="C1", alpha=0.13, zorder=1,
              label="range the search tried to match")
    a.plot(r[near], (w * theory)[near], color="k", lw=2.0, ls="--", zorder=3,
           label="theory")
    a.plot(r[near], (w * xi_zoom[best])[near], "o-", color="C0", ms=5, lw=1.3,
           zorder=4, label=f"closest of {NREAL} seeds")
    a.axhline(0.0, color="0.5", lw=0.5)
    a.set_xlabel(r"$r$ [Mpc/$h$]")
    a.set_ylabel(r"$r^2\,\xi(r)$ [(Mpc/$h$)$^2$]")
    a.set_title(f"(b) the best seed out of {NREAL} still misses", fontsize=11)
    a.text(0.03, 0.04,
           f"off by {float(np.sqrt((dev_best ** 2).mean())):.2f} of the "
           "scatter, on average;\na code reporting $\\xi$ "
           f"{100 * (1 - BROKEN):.0f}% low would move it by only "
           f"{(1 - BROKEN) * abs(theory[j65]) / fair_std[j65]:.2f}",
           transform=a.transAxes, fontsize=10, color="C0")
    a.legend(frameon=False, fontsize=9, loc="upper right")

    a = ax[1, 0]
    a.plot(100 * fracs, damage, "o-", color="C3", ms=6, lw=1.6, zorder=3,
           label=r"seeds ranked by how well $\xi$ matches")
    a.plot(100 * dens_cut[0], dens_cut[1], "D", color="C2", ms=9, zorder=4,
           label="seeds rejected on the density of\nthe subvolume alone")
    a.set_xscale("log")
    a.set_xticks([5, 10, 20, 40, 60, 80, 100])
    a.set_xticklabels(["5", "10", "20", "40", "60", "80", "100"])
    a.invert_xaxis()
    a.axhline(1.0, color="k", lw=1.0, ls="--")
    a.set_ylim(0.0, 1.25)
    a.set_xlabel("percent of seeds accepted  (tighter match to the right)")
    a.set_ylabel("scatter kept, over the matched range")
    a.set_title("(c) matching on $\\xi$ removes scatter at any tolerance",
                fontsize=11)
    a.text(0.04, 0.10,
           f"rejecting only the worst {100 - 100 * xi_cut[0]:.0f}% of seeds\n"
           f"already removes {100 - 100 * xi_cut[1]:.0f}% of the scatter;\n"
           "a cut on subvolume density alone removes none",
           transform=a.transAxes, fontsize=10, color="C3")
    a.legend(frameon=False, fontsize=9, loc="upper left")

    a = ax[1, 1]
    jout = int(np.argmin(np.abs(r - 45.0)))
    bins = np.linspace(xi_zoom[:, jout].min(), xi_zoom[:, jout].max(), 26)
    a.hist(xi_zoom[:, jout], bins=bins, color="0.75", label="all seeds")
    a.hist(xi_zoom[kept, jout], bins=bins, color="C3", alpha=0.8,
           label=f"kept seeds ({kept.sum()})")
    a.axvline(fair_mean[jout], color="0.35", lw=1.6)
    a.axvline(sel_mean[jout], color="C3", lw=1.6)
    a.annotate("", xy=(sel_mean[jout], 0.80 * a.get_ylim()[1]),
               xytext=(fair_mean[jout], 0.80 * a.get_ylim()[1]),
               arrowprops=dict(arrowstyle="->", color="C3", lw=1.4))
    a.set_xlabel(rf"$\xi$ at $r = {r[jout]:.0f}$ Mpc/$h$")
    a.set_ylabel("number of seeds")
    a.set_title(f"(d) a separation the search never looked at", fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="upper right")

    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.4)

    fig.suptitle(f"{NREAL} realizations of a "
                 f"{NZOOM * CELL:.0f}$\\times${NZOOM * CELL:.0f}$\\times$"
                 f"{LBOX:.0f} Mpc/$h$ subvolume inside a {LBOX:.0f} Mpc/$h$ box",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig("seed_conditioning.pdf", bbox_inches="tight")
    fig.savefig("seed_conditioning.png", dpi=150, bbox_inches="tight")
    print("\nSaved: seed_conditioning.pdf, seed_conditioning.png")


if __name__ == "__main__":
    main()
