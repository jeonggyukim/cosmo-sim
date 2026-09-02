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
ap.add_argument("--nboot", type=int, default=500)
ap.add_argument("--nnull", type=int, default=200)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "selection_summary.png"))
A = ap.parse_args()

with h5py.File(f"{A.data}/theory.hdf5") as f:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = names.index("matter")
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    kny, dkperp, N, L = (f.attrs[x] for x in ("kny", "dkperp", "N", "L"))
lo = k <= 2*dkperp

P, COLS, RS, seed_of = [], {}, None, []
for si, fn in enumerate(sorted(glob.glob(f"{A.data}/seed_*/pk.hdf5"))):
    with h5py.File(fn) as f:
        p = f["P_pencil"][SP]
        P.append(p)
        RS = f["smooth_R"][:]
        for r, R in enumerate(RS):
            COLS.setdefault(f"tidal shear\nR={R:g}", []).append(f["shear"][r])
            COLS.setdefault(f"mean overdensity\nR={R:g}", []).append(f["dbar"][r])
            for w, wn in enumerate(("knot", "filament", "sheet", "void")):
                COLS.setdefault(f"{wn} fraction\nR={R:g}", []).append(f["webtype"][r][:, w])
        seed_of.append(np.full(p.shape[0], si))
P = np.concatenate(P)                                   # (nmeas, nk)
COLS = {n: np.concatenate(v) for n, v in COLS.items()}
seed_of = np.concatenate(seed_of)
nseed = seed_of.max() + 1
by_seed = [np.where(seed_of == s)[0] for s in range(nseed)]

# Two criteria, differing only in the curve the pencil is asked to match.
crit_th = np.sqrt((np.log(P[:, lo]/P_th[lo])**2).mean(1))
crit_wn = np.sqrt((np.log(P[:, lo]/P_win[lo])**2).mean(1))
nkeep = max(1, int(round(A.keep*len(crit_th))))
keep_th = np.argsort(crit_th)[:nkeep]
keep_wn = np.argsort(crit_wn)[:nkeep]

# Quantities worth showing: the ones the criterion never mentions, plus the
# smoothing radius that shows the effect most clearly.
SHOW = [n for n in COLS if n.endswith("R=20")] + [n for n in COLS if n.endswith("R=40")]
SHOW = [n for n in SHOW if not n.startswith("mean overdensity")] + \
       [n for n in SHOW if n.startswith("mean overdensity")]


def shift(c, T, idx=None):
    if idx is not None:
        c, T = c[idx], T[idx]
    kp = np.argsort(c)[:max(1, int(round(A.keep*len(c))))]
    return (T[kp].mean() - T.mean())/T.std()


rng = np.random.default_rng(1)
res = {}
for name in SHOW:
    T = COLS[name]
    row = {}
    for tag, c in (("theory", crit_th), ("window", crit_wn)):
        pt = shift(c, T)
        boot = np.empty(A.nboot)
        for b in range(A.nboot):
            idx = np.concatenate([by_seed[s] for s in rng.integers(0, nseed, nseed)])
            boot[b] = shift(c, T, idx)
        row[tag] = (pt, np.percentile(boot, [16, 84]))
    nulls = np.array([shift(rng.permutation(crit_th), T) for _ in range(A.nnull)])
    row["null"] = (nulls.mean(), np.percentile(nulls, [16, 84]))
    res[name] = row

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 7.2),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))

# ---- left: the spectra, as ratios to the linear theory -----------------------
axL.axhline(1.0, color="0.25", lw=2.0, label="linear theory  $P_{\\rm th}$")
axL.plot(k, P_win/P_th, color="C0", lw=2.2,
         label="theory $\\ast$ pencil window\n(what the estimator measures)")
axL.plot(k, P.mean(0)/P_th, color="C0", ls="--", lw=1.6,
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
axL.set_title("The pencil measures a deficit that is geometric,\n"
              "and selection climbs out of it", fontsize=10.5)

# ---- right: shifts in quantities the criterion never mentions ----------------
y = np.arange(len(SHOW))[::-1]
style = [("null", "0.6", "o", "random criterion (noise floor)"),
         ("window", "C2", "s", "match theory $\\ast$ window (control)"),
         ("theory", "C3", "D", "match raw theory (the proposal)")]
for off, (tag, col, mk, lab) in zip((+0.26, 0.0, -0.26), style):
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
axR.set_xlim(-1.12, 1.28)
axR.text(1.0, len(SHOW) - 0.4, " 1 sd of the scatter\n between regions",
         fontsize=8, color="0.4", va="top", ha="left")
axR.set_yticks(y)
axR.set_yticklabels([n.replace("\n", " ") for n in SHOW], fontsize=8.5)
axR.set_xlabel("shift of the selected regions  [standard deviations]")
axR.legend(fontsize=9, loc="lower right", framealpha=0.95)
axR.set_title("Selecting on low-$k$ power moves the region's structure,\n"
              "but only when the target curve is the wrong one", fontsize=10.5)
axR.grid(axis="x", alpha=0.25)

fig.suptitle(f"Selecting a pencil subvolume on its power spectrum: "
             f"{nseed} realizations, $N={int(N)}^3$, $L={L:g}$ Mpc/$h$, 2LPT, "
             f"$\\delta(q)$ matter, pencil = $(L/8)^2\\times L$, keeping {100*A.keep:g}%",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(A.out, dpi=300)
print(f"wrote {A.out}")
for n in SHOW:
    r = res[n]
    print(f"{n.replace(chr(10), ' '):<28} theory {r['theory'][0]:+.3f}  "
          f"window {r['window'][0]:+.3f}  null {r['null'][0]:+.3f}")
