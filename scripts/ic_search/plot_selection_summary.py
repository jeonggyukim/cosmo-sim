#!/usr/bin/env python3
"""Summary figure: what selecting on the pencil spectrum does, and against what.

Left panel shows the spectra. A pencil measures a spectrum well below the linear
theory at low k, deterministically, because masking a region convolves its power
spectrum with the window of that region. The ensemble mean of the measurements
lies on the convolved curve, which is the curve the estimator is unbiased for.
The pencils selected for matching the *unconvolved* theory lie somewhere else
entirely: they have been pushed up onto a curve their own mean is nowhere near.

Right panel shows what that does to the region, in quantities the selection
criterion never looks at. Three criteria are compared: matching the raw theory,
matching the convolved theory, and a randomly permuted criterion that selects
nothing in particular. The random one fixes the noise floor of the whole
procedure; the convolved one is the control that asks for nothing atypical.

Usage:
    python plot_selection_summary.py [--data DIR] [--keep 0.05] [--out PNG]
"""
import argparse, glob, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["savefig.dpi"] = 300
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "web_n128_all"))
ap.add_argument("--keep", type=float, default=0.05)
ap.add_argument("--keep2", type=float, default=0.001,
                help="a second, tighter cut, drawn beside the first so the\n"
                     "saturation is visible in the figure rather than asserted")
ap.add_argument("--nboot", type=int, default=500)
ap.add_argument("--nnull", type=int, default=200)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "selection_summary.png"))
ap.add_argument("--cache", default=None,
                help="npz holding everything the figure draws. Default is the "
                     "output path with a .npz suffix. Reading 800 chunk files and "
                     "running 20000 resamples takes minutes; redrawing from the "
                     "cache takes a second, which is what a change of colour needs")
ap.add_argument("--recompute", action="store_true",
                help="ignore the cache even when its signature matches")
A = ap.parse_args()
CACHE = A.cache or os.path.splitext(A.out)[0] + ".npz"

# The theory curves and the geometry come from chunkio, which reads them from a
# chunk file when there is one and from theory.hdf5 for the per-seed layout.
import chunkio

import glob as _glob
# The signature is everything that changes the numbers. Anything else -- colours,
# labels, limits -- redraws from the cache without touching the data.
SIG = np.array([len(_glob.glob(f"{A.data}/chunk_*.hdf5")), A.keep, A.keep2,
                A.nboot, A.nnull], float)
CACHED = None
if os.path.exists(CACHE) and not A.recompute:
    _z = np.load(CACHE, allow_pickle=True)
    if _z["sig"].shape == SIG.shape and np.allclose(_z["sig"], SIG):
        CACHED = _z
        print(f"redrawing from {CACHE}")

crit_th, crit_wn, COLS, meta, P = chunkio.load(A.data, "matter", want_spectra=True)
nseed, npen = crit_th.shape
k, P_th, P_win, lo = meta["k"], meta["P_th"], meta["P_win"], meta["lo"]
N, L, kny = meta["N"], meta["L"], meta["kny"]
P = P.reshape(-1, P.shape[-1])
COLS = {n: v.ravel() for n, v in COLS.items()}
crit_th, crit_wn = crit_th.ravel(), crit_wn.ravel()
nkeep = max(1, int(round(A.keep*len(crit_th))))
keep_th = np.argsort(crit_th)[:nkeep]
keep_wn = np.argsort(crit_wn)[:nkeep]

PREFER = ["tidal shear", "knot fraction", "filament fraction", "sheet fraction",
          "void fraction", "bulk flow", "mean overdensity"]
_keep = chunkio.usable(COLS)
SHOW = [n for stem in PREFER for n in COLS if n.startswith(stem) and _keep(n)][:12]

def shift(c, T, idx=None, keep=None):
    if idx is not None:
        c, T = c[idx], T[idx]
    n = max(1, int(round((A.keep if keep is None else keep)*len(c))))
    kp = np.argpartition(c, n)[:n]   # partial: the order inside does not matter
    return (T[kp].mean() - T.mean())/T.std()


# Predicted shift from the conditioning identity: a quantity correlated at rho
# with the one being selected on moves by rho times its shift, and nothing more.
pw = COLS["large-scale power"]
shift_power = shift(crit_th, pw)
RHO = {n: np.corrcoef(pw, COLS[n])[0, 1] for n in SHOW}

rng = np.random.default_rng(1)
res = {}
for name in SHOW:
    T = COLS[name]
    row = {}
    for tag, c in (("theory", crit_th), ("window", crit_wn)):
        pt = shift(c, T)
        # Resample whole realizations, not pencils: the 24 pencils of one box
        # share that box's modes and are not independent draws.
        boot = np.empty(A.nboot)
        for b in range(A.nboot):
            g = rng.integers(0, nseed, nseed)
            idx = (g[:, None]*npen + np.arange(npen)).ravel()
            boot[b] = shift(c, T, idx)
        row[tag] = (pt, np.percentile(boot, [16, 84]))
    nulls = np.array([shift(rng.permutation(crit_th), T) for _ in range(A.nnull)])
    row["null"] = (nulls.mean(), np.percentile(nulls, [16, 84]))
    # The same criterion applied ten times harder. Its near-coincidence with the
    # looser cut is the saturation: the shift is bounded by the scatter between
    # regions, not by how many seeds are searched.
    pt2 = shift(crit_th, T, keep=A.keep2)
    boot2 = np.empty(A.nboot)
    for b in range(A.nboot):
        g = rng.integers(0, nseed, nseed)
        idx = (g[:, None]*npen + np.arange(npen)).ravel()
        boot2[b] = shift(crit_th, T, idx, keep=A.keep2)
    row["theory2"] = (pt2, np.percentile(boot2, [16, 84]))
    res[name] = row

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 7.2),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))

# ---- left: the spectra, as ratios to the linear theory -----------------------
axL.axhline(1.0, color="0.25", lw=2.0, label="linear theory  $P_{\\rm th}$")
# The same realizations measured without a mask. Drawn thick and faint beneath
# everything else: it is the reference the pencil curves depart from, and its
# flatness at 1 is what says the departure belongs to the window and not to the
# estimator.
if meta.get("P_box") is not None:
    # Magenta rather than grey: the panel already uses grey for the theory line
    # and for the shaded criterion band, so a third grey was unreadable.
    axL.plot(k, meta["P_box"].mean(0)/P_th, color="magenta", lw=6.0, alpha=0.35,
             solid_capstyle="round", zorder=1,
             label=f"whole box, mean of {len(meta['P_box']):,} seeds")
# Drawn thick and pale, with the measured mean laid over it in a different
# colour. The two coincide, which is the point: the prediction is underneath and
# the measurement sits on it. Giving both the same colour hid the measurement
# entirely and made the agreement impossible to see.
axL.plot(k, P_win/P_th, color="C0", lw=5.0, alpha=0.40, solid_capstyle="round",
         zorder=2,
         label="theory $\\ast$ pencil window\n(what the estimator measures)")
axL.plot(k, P.mean(0)/P_th, color="k", ls="--", lw=1.4, zorder=3,
         label=f"mean of all {len(P):,} pencils")
axL.plot(k, P[keep_th].mean(0)/P_th, color="C3", lw=2.2,
         label=f"mean of the {100*A.keep:g}% closest to $P_{{\\rm th}}$")
axL.plot(k, P[keep_wn].mean(0)/P_th, color="C2", lw=2.2, ls=":",
         label=f"mean of the {100*A.keep:g}% closest to theory $\\ast$ window")
axL.axvspan(k[0], k[lo][-1], color="0.9", zorder=0)
axL.text(k[0]*1.05, 0.35, "band used\nby the criterion", fontsize=9, color="0.4")
axL.set_xscale("log")
# The default log minor labels collide at this width; label a chosen few instead.
axL.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]))
axL.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axL.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axL.set_xlabel("$k$  [$h$ Mpc$^{-1}$]")
axL.set_ylabel("$P(k)\\, /\\, P_{\\rm theory}(k)$")
axL.set_ylim(0.3, 1.35)
axL.set_xlim(k[0], 0.9*kny)
axL.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
axL.set_title("A small region always measures less power than the theory,\n"
              "and the search picks regions that do not", fontsize=10.5)

# ---- right: shifts in quantities the criterion never mentions ----------------
y = np.arange(len(SHOW))[::-1]
style = [("null", "0.6", "o", "random criterion (noise floor)"),
         ("window", "C2", "s", "match theory $\\ast$ window (control)"),
         ("theory", "C3", "D", f"match raw theory, keeping {100*A.keep:g}%"),
         ("theory2", "C1", "v", f"match raw theory, keeping {100*A.keep2:g}%")]
axR.plot([RHO[n]*shift_power for n in SHOW], y - 0.30, marker="|", ls="none",
         ms=13, mew=1.8, color="0.15", zorder=5,
         label=("predicted: $\\rho$ with the selected power $\\times$ its own"
                "\nshift of %.2f$\\sigma$" % shift_power))
for off, (tag, col, mk, lab) in zip((+0.30, +0.10, -0.10, -0.30), style):
    v = np.array([res[n][tag][0] for n in SHOW])
    e = np.array([res[n][tag][1] for n in SHOW])
    axR.errorbar(v, y + off, xerr=np.abs(e.T - v), fmt=mk, color=col, ms=5,
                 lw=1.4, capsize=2.5, label=lab)
axR.axvline(0.0, color="0.3", lw=1.0)
# Without a reference the axis runs to about half a standard deviation and makes
# these shifts look like the whole distribution. Mark where one standard
# deviation of the region-to-region scatter actually is.
for x in (-1.0, 1.0):
    axR.axvline(x, color="0.45", lw=1.0, ls=":")
# Extended past the 1 sd mark on the right to leave the legend somewhere it does
# not cover the points, now that there are four series and a prediction marker.
axR.set_xlim(-1.12, 1.85)
axR.text(1.0, len(SHOW) - 0.4, " 1 sd of the scatter\n between regions",
         fontsize=8, color="0.4", va="top", ha="left")
axR.set_yticks(y)
axR.set_yticklabels([n.replace("\n", " ") for n in SHOW], fontsize=8.5)
axR.set_xlabel(f"shift {chunkio.SHIFT_SYMBOL} of the selected regions"
               "  [standard deviations]")
axR.legend(fontsize=9, loc="lower right", framealpha=0.95)
axR.set_title("The criterion never measures these, and moves them anyway,\n"
              "by their correlation with the power it does measure",
              fontsize=10.5)
axR.grid(axis="x", alpha=0.25)

fig.suptitle(f"Selecting a pencil subvolume on its power spectrum: "
             f"{nseed} realizations, $N={int(N)}^3$, $L={L:g}$ Mpc/$h$, 2LPT, "
             f"$\\delta(q)$ matter, pencil = $(L/8)^2\\times L$, keeping {100*A.keep:g}%",
             fontsize=11)
# Reserve a strip at the bottom for the two definition lines. Without it they
# are drawn over the tick labels and the axis label, which is where they landed.
fig.tight_layout(rect=(0, 0.075, 1, 0.94))
# After the layout, so the panel positions are final, and under the right panel,
# which is the one whose axis is the shift.
chunkio.annotate_shift(fig, ax=axR, y=0.012, fontsize=9.5)
fig.savefig(A.out, dpi=300)
print(f"wrote {A.out}")
for n in SHOW:
    r = res[n]
    print(f"{n.replace(chr(10), ' '):<28} theory {r['theory'][0]:+.3f}  "
          f"window {r['window'][0]:+.3f}  null {r['null'][0]:+.3f}")
