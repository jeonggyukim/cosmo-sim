#!/usr/bin/env python3
"""Distribution of each region property, before and after selection.

One panel per quantity. The filled grey histogram is every pencil measured. The
two outlined histograms are the pencils a selection keeps: red for the proposed
criterion, which asks the pencil to match the raw linear theory, and green for
the same selection against the theory convolved with the pencil window.

The summary figure reports each of these as a single number, the shift of the
mean in units of the population scatter. That number hides whether the kept
sample is a displaced copy of the parent distribution or a narrowed piece of it,
and the two mean different things for a simulation drawn from it. These panels
show which is happening.

Usage:
    python plot_selection_histograms.py [--data DIR] [--keep 0.05] [--out PNG]
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
ap.add_argument("--out", default=os.path.join(paths.FIGS, "selection_histograms.png"))
A = ap.parse_args()

# The theory curves and the geometry come from chunkio, which reads them from a
# chunk file when there is one and from theory.hdf5 for the per-seed layout.


def shape_params(lam):
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    Lnorm = np.sqrt((lam**2).sum(-1)) + 1e-30
    return (l1 - l3)/(2*Lnorm), (l1 - 2*l2 + l3)/(2*Lnorm)


import chunkio

crit_th, crit_wn, C, meta, _ = chunkio.load(A.data, "matter")
nseed, npen = crit_th.shape
C = {n: v.ravel() for n, v in C.items()}
crit_th, crit_wn = crit_th.ravel(), crit_wn.ravel()
N, L = meta["N"], meta["L"]
nk = max(1, int(round(A.keep*len(crit_th))))
kth, kwn = np.argsort(crit_th)[:nk], np.argsort(crit_wn)[:nk]

PREFER = ["large-scale power", "small-scale power", "tidal shear", "ellipticity",
          "knot fraction", "filament fraction", "sheet fraction", "void fraction",
          "bulk flow", "env contrast", "mean overdensity"]
SHOW = [n for stem in PREFER for n in C if n.startswith(stem)][:12]

ncol = 4
nrow = int(np.ceil(len(SHOW)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 2.9*nrow))
axes = np.atleast_1d(axes).ravel()

for ax, name in zip(axes, SHOW):
    T = C[name]
    sd = T.std()
    bins = np.linspace(*np.percentile(T, [0.2, 99.8]), 46)
    ax.hist(T, bins=bins, color="0.78", edgecolor="none",
            label=f"all {len(T):,}", density=True)
    for idx, col, lab in ((kwn, "C2", "match theory $\\ast$ window"),
                          (kth, "C3", "match raw theory")):
        ax.hist(T[idx], bins=bins, histtype="step", lw=1.7, color=col, density=True,
                label=lab)
        ax.axvline(T[idx].mean(), color=col, lw=1.1, ls="--")
    ax.axvline(T.mean(), color="0.35", lw=1.1)
    sh_th = (T[kth].mean() - T.mean())/sd
    sh_wn = (T[kwn].mean() - T.mean())/sd
    ax.set_title(name, fontsize=9.5)
    # Put the label on whichever side carries less of the histogram, so it does
    # not land on a peak. Several of these distributions are strongly one-sided.
    counts, _ = np.histogram(T, bins=bins)
    third = max(1, len(counts)//3)
    left = counts[:third].sum() > counts[-third:].sum()
    ax.text(0.97 if left else 0.03, 0.96,
            f"shift  {sh_th:+.2f}$\\sigma$ / {sh_wn:+.2f}$\\sigma$",
            transform=ax.transAxes, fontsize=8, va="top",
            ha="right" if left else "left",
            bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=1.8))
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    # Several of these quantities are of order 1e-4, and the default tick labels
    # then overlap; four ticks with a shared exponent fit.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(7.5)

for ax in axes[len(SHOW):]:
    ax.axis("off")
# A per-panel legend lands on the data in whichever panel it is put, so it goes
# at figure level instead.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.899),
           ncol=3, fontsize=9, frameon=False)

fig.suptitle(f"Region properties before and after keeping the closest {100*A.keep:g}% of "
             f"pencils\n{nseed} realizations, $N={int(N)}^3$, $L={L:g}$ Mpc/$h$, 2LPT, "
             f"$\\delta(q)$ matter, pencil $=(L/8)^2\\times L$",
             fontsize=11, y=0.985)
fig.text(0.5, 0.917, "Dashed lines mark the means. The label in each panel gives the shift "
         "of the mean in units of the scatter over all pencils, as raw / window.",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0, 1, 0.858))
fig.savefig(A.out, dpi=300)
print(f"wrote {A.out}")
for name in SHOW:
    T = C[name]
    print(f"{name:<30} shift raw {(T[kth].mean()-T.mean())/T.std():+.3f}  "
          f"window {(T[kwn].mean()-T.mean())/T.std():+.3f}  "
          f"width kept/all {T[kth].std()/T.std():.2f}")
