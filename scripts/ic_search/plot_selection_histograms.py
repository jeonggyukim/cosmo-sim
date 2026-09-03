#!/usr/bin/env python3
"""Distribution of each region property, before and after selection.

One panel per quantity. The filled grey histogram is every pencil measured. The
outlined histograms are the pencils a selection keeps: red and orange for the
proposed criterion, which asks the pencil to match the raw linear theory, at two
cuts a factor of ten apart, and green for the same selection against the theory
convolved with the pencil window.

The summary figure reports each of these as a single number, the shift of the
mean in units of the population scatter. That number hides whether the kept
sample is a displaced copy of the parent distribution or a narrowed piece of it,
and the two mean different things for a simulation drawn from it. These panels
show which is happening.

The two cuts show the saturation directly: a tenfold tighter selection moves the
outlined histogram very little, because the criterion selects on a noisy proxy
for these quantities rather than on the quantities themselves.

Usage:
    python plot_selection_histograms.py [--data DIR] [--keep 0.01]
                                        [--keep2 0.001] [--out PNG]
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
ap.add_argument("--keep", type=float, default=0.01)
ap.add_argument("--keep2", type=float, default=0.001,
                help="a second, tighter cut on the same criterion, so the "
                     "figure shows how little a tenfold tighter search moves "
                     "the kept distribution")
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
order_th = np.argsort(crit_th)
nk = max(1, int(round(A.keep*len(crit_th))))
nk2 = max(1, int(round(A.keep2*len(crit_th))))
kth, kth2 = order_th[:nk], order_th[:nk2]
kwn = np.argsort(crit_wn)[:nk]

# Every stem ends in " R" so that only the pencil measurements match: the
# interior and whole-box variants are named "... interior R=" and "... box R=",
# and a panel is better spent on another quantity than on a second view of one
# already shown. The whole-box variants are in the summary figure, where the
# comparison between the region and its parent box is the point.
# Ellipticity is left out. With the normalisation of shape_params it is
# (l1-l3)/2|l|, which is at most 1/sqrt(2) = 0.7071, reached when l2 = 0 and
# l1 = -l3. Smoothed over a region the measured values sit within 0.003 of that
# ceiling, so the histogram is a spike against a wall and its shift measures how
# close a sample gets to the bound rather than anything about the region.
PREFER = ["large-scale power", "small-scale power", "tidal shear R",
          "knot fraction R", "filament fraction R", "sheet fraction R",
          "void fraction R", "mean overdensity R", "bulk flow", "env contrast R"]
_keep = chunkio.usable(C)
SHOW = [n for stem in PREFER for n in C if n.startswith(stem) and _keep(n)][:16]

ncol = 4
nrow = int(np.ceil(len(SHOW)/ncol))
FIGH = 2.9*nrow + 1.6      # the panels, plus a strip for title, caption, legend
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0*ncol, FIGH))
axes = np.atleast_1d(axes).ravel()

LAB_WN = f"match theory $\\ast$ window, keeping {100*A.keep:g}%"
LAB_TH = f"match raw theory, keeping {100*A.keep:g}%"
LAB_TH2 = f"match raw theory, keeping {100*A.keep2:g}%"

for ax, name in zip(axes, SHOW):
    T = C[name]
    sd = T.std()
    bins = np.linspace(*np.percentile(T, [0.2, 99.8]), 46)
    # The tighter cut keeps a tenth as many pencils, so the same 45 bins would
    # scatter it beyond reading. Every other edge halves the count per bin
    # without changing what a density histogram is comparable to.
    bins2 = bins[::2]
    ax.hist(T, bins=bins, color="0.78", edgecolor="none",
            label=f"all {len(T):,}", density=True)
    for idx, bb, col, lab in ((kwn, bins, "C2", LAB_WN),
                              (kth, bins, "C3", LAB_TH),
                              (kth2, bins2, "C1", LAB_TH2)):
        ax.hist(T[idx], bins=bb, histtype="step", lw=1.7, color=col, density=True,
                label=lab)
        ax.axvline(T[idx].mean(), color=col, lw=1.1, ls="--")
    ax.axvline(T.mean(), color="0.35", lw=1.1)
    ax.set_title(name, fontsize=9.5)
    # Put the label on whichever side carries less of the histogram, so it does
    # not land on a peak. Several of these distributions are strongly one-sided.
    counts, _ = np.histogram(T, bins=bins)
    third = max(1, len(counts)//3)
    left = counts[:third].sum() > counts[-third:].sum()
    xt, ha = (0.97, "right") if left else (0.03, "left")
    # One line per selection, each in its own colour, so the reader does not
    # have to match three numbers on one line to three curves by their order.
    for j, (idx, col) in enumerate(((kth, "C3"), (kth2, "C1"), (kwn, "C2"))):
        ax.text(xt, 0.96 - 0.105*j,
                f"{chunkio.SHIFT_SYMBOL} = {(T[idx].mean() - T.mean())/sd:+.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha=ha, color=col,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
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


def _y(inches_from_top):
    """Place header text a fixed distance below the top edge, not a fixed
    fraction of the figure: the number of rows sets the height, so a fraction
    that suits three rows crowds four."""
    return 1 - inches_from_top/FIGH


fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, _y(1.15)),
           ncol=4, fontsize=9, frameon=False)

fig.suptitle(f"Region properties before and after selection, at two cuts a factor "
             f"of ten apart\n{nseed:,} realizations, $N={int(N)}^3$, $L={L:g}$ Mpc/$h$, "
             f"2LPT, $\\delta(q)$ matter, pencil $=(L/8)^2\\times L$",
             fontsize=11, y=_y(0.17))
fig.text(0.5, _y(0.92), f"Dashed lines mark the means. {chunkio.SHIFT_DEF} is the shift "
         f"of the mean in units of the scatter over all pencils, coloured to match its "
         f"curve. Tightening the cut from {100*A.keep:g}% to {100*A.keep2:g}% barely "
         f"moves it.",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0, 1, _y(1.45)))
fig.savefig(A.out, dpi=300)
print(f"wrote {A.out}")
print(f"kept {nk:,} of {len(crit_th):,} at {100*A.keep:g}%, "
      f"{nk2:,} at {100*A.keep2:g}%\n")
hdr = f"{'quantity':<30}{'raw':>8}{'raw tight':>11}{'window':>9}{'width':>8}"
print(hdr); print("-"*len(hdr))
for name in SHOW:
    T = C[name]
    sd = T.std()
    print(f"{name:<30}{(T[kth].mean()-T.mean())/sd:>+8.3f}"
          f"{(T[kth2].mean()-T.mean())/sd:>+11.3f}"
          f"{(T[kwn].mean()-T.mean())/sd:>+9.3f}"
          f"{T[kth].std()/sd:>8.2f}")
