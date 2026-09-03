#!/usr/bin/env python3
"""The same quantity measured in a pencil and in the box that contains it.

region_vs_box reports the two as separate points on a shared axis, each divided
by its own scatter. That is two different units on one axis: the box-to-box
scatter is a fifth of the region-to-region scatter for the shear and a
twentieth for the web fractions, so a box point drawn beside a region point
reads as a far larger share of the region's displacement than it is.

Plotting them against each other removes the ambiguity, because both axes carry
the same quantity and can therefore carry the same unit. Both are divided by the
PENCIL scatter, so the box distribution appears as the narrow band it is, and
the vertical displacement of the selected sample reads directly as the fraction
of the region's shift that the box inherited.

Every pencil is drawn, which repeats each box value once per pencil of that box.
The repetition leaves a normalised density exactly unchanged and keeps the
selected samples the same size as in every other figure; it shows only as
horizontal banding among the plotted points.

Usage:
    python plot_pencil_vs_box.py --data DIR [--q "tidal shear R=9"]
                                 [--keep 0.01] [--out PNG]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["savefig.dpi"] = 300
import paths, chunkio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"))
ap.add_argument("--species", default="matter")
ap.add_argument("--q", default="tidal shear R=9")
ap.add_argument("--keep", type=float, default=0.01)
ap.add_argument("--keep2", type=float, default=0.001)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "pencil_vs_box.png"))
A = ap.parse_args()

crit_th, crit_wn, C, meta, _ = chunkio.load(A.data, A.species)
nseed, npen = crit_th.shape
BOXQ = A.q.replace(" R=", " box R=")
for n in (A.q, BOXQ):
    if n not in C:
        raise SystemExit(f"no quantity named {n!r}. Available: "
                         + ", ".join(sorted(C)))

# Every pencil, and every pencil of one box carries that box's value repeated.
# The repetition is harmless here: replicating each box value the same number of
# times leaves a normalised density exactly unchanged, and it keeps the selected
# samples the same size as in every other figure. It shows only as horizontal
# banding among the plotted points, which is what the data is.
P = C[A.q].ravel()
B = C[BOXQ].ravel()
cth = crit_th.ravel()
cwn = crit_wn.ravel()
npoint = len(P)

# Both axes in units of the PENCIL scatter. Standardising each to its own would
# rescale the box axis by 1/0.2 and hide exactly what the figure is for.
sd = P.std()
x = (P - P.mean())/sd
y = (B - B.mean())/sd
sd_ratio = B.std()/sd
rho = float(np.corrcoef(x, y)[0, 1])

nk = max(1, int(round(A.keep*npoint)))
nk2 = max(1, int(round(A.keep2*npoint)))
o = np.argsort(cth)
SELECTIONS = [
    (np.argsort(cwn)[:nk], "C2", 4.0, 0.35,
     f"match theory $\\ast$ window, {100*A.keep:g}%  (control)"),
    (o[:nk], "C3", 4.0, 0.35, f"match raw theory, {100*A.keep:g}%"),
    (o[:nk2], "C1", 13.0, 0.90, f"match raw theory, {100*A.keep2:g}%"),
]

fig = plt.figure(figsize=(9.0, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=(4.4, 1.15), height_ratios=(1.15, 4.4),
                      wspace=0.045, hspace=0.045,
                      left=0.100, right=0.977, bottom=0.115, top=0.775)
ax = fig.add_subplot(gs[1, 0])
axx = fig.add_subplot(gs[0, 0], sharex=ax)
axy = fig.add_subplot(gs[1, 1], sharey=ax)

lo, hi = np.percentile(x, [0.05, 99.95])
for idx, *_ in SELECTIONS:
    lo, hi = min(lo, x[idx].min()), max(hi, x[idx].max())
pad = 0.05*(hi - lo)
XL = (lo - pad, hi + pad)
# The same limits on both axes. Different ones would restore the distortion this
# figure exists to remove: equal visual lengths must mean equal displacements.
YL = XL

H, xe, ye = np.histogram2d(x, y, bins=150, range=(XL, YL))
from scipy.ndimage import gaussian_filter
H = gaussian_filter(H, 1.4)
flat = np.sort(H.ravel())[::-1]
cum = np.cumsum(flat)/H.sum()
levels = [flat[np.searchsorted(cum, f)] for f in (0.99, 0.9, 0.68, 0.38)]
xc, yc = 0.5*(xe[1:] + xe[:-1]), 0.5*(ye[1:] + ye[:-1])
ax.contourf(xc, yc, H.T, levels=levels + [H.max()],
            colors=["0.90", "0.80", "0.68", "0.55"], zorder=0)
ax.contour(xc, yc, H.T, levels=levels, colors="0.45", linewidths=0.5, zorder=1)

for j, (idx, col, size, al, lab) in enumerate(SELECTIONS):
    ax.scatter(x[idx], y[idx], s=size, marker="o", c=col, alpha=al,
               linewidths=0, zorder=3 + j, label=lab)
for idx, col, _, _, _ in SELECTIONS:
    ax.plot([x[idx].mean()], [y[idx].mean()], marker="P", ms=13, mfc=col,
            mec="k", mew=1.2, zorder=8)

ax.plot(XL, XL, color="k", lw=1.3, ls=":", zorder=2,
        label="equal displacement")
ax.axhline(0, color="0.35", lw=0.9, zorder=2)
ax.axvline(0, color="0.35", lw=0.9, zorder=2)
ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_aspect("equal")
ax.set_xlabel(f"{A.q}, measured in the pencil   "
              "[units of the pencil scatter]")
ax.set_ylabel(f"{A.q}, measured in the whole box\n"
              "[the same units]")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

bx = np.linspace(*XL, 70)
axx.hist(x, bins=bx, color="0.78", edgecolor="none", density=True)
axy.hist(y, bins=bx, color="0.78", edgecolor="none", density=True,
         orientation="horizontal")
for idx, col, _, _, _ in SELECTIONS:
    step = 1 if len(idx) == nk else 2
    axx.hist(x[idx], bins=bx[::step], histtype="step", lw=1.6, color=col,
             density=True)
    axy.hist(y[idx], bins=bx[::step], histtype="step", lw=1.6, color=col,
             density=True, orientation="horizontal")
    axx.axvline(x[idx].mean(), color=col, lw=1.1, ls="--")
    axy.axhline(y[idx].mean(), color=col, lw=1.1, ls="--")
for a, is_x in ((axx, True), (axy, False)):
    a.set_yticks([]) if is_x else a.set_xticks([])
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
axx.spines["left"].set_visible(False)
axy.spines["bottom"].set_visible(False)
plt.setp(axx.get_xticklabels(), visible=False)
plt.setp(axy.get_yticklabels(), visible=False)

rows_txt = [f"$\\sigma_{{\\rm box}}/\\sigma_{{\\rm pencil}} = {sd_ratio:.3f}$,"
            f"   $\\rho = {rho:+.3f}$"]
for idx, col, _, _, lab in SELECTIONS:
    dx, dy = x[idx].mean(), y[idx].mean()
    frac = f"{dy/dx:>6.1%}" if abs(dx) > 1e-3 else "     -"
    rows_txt.append(f"{lab.split(',')[0]}:  pencil ${dx:+.2f}$,  "
                    f"box ${dy:+.3f}$,  inherited {frac}")
ax.text(0.985, 0.022, "\n".join(rows_txt), transform=ax.transAxes, fontsize=8.6,
        va="bottom", ha="right", color="0.15", linespacing=1.5, zorder=9,
        bbox=dict(fc="white", ec="0.75", alpha=0.94, pad=4.5))

fig.suptitle(f"{nseed:,} realizations $\\times$ {npen} pencils, "
             f"$N={int(meta['N'])}^3$, $L={meta['L']:g}$ Mpc/$h$, "
             f"2LPT, $\\delta(q)$ matter", fontsize=11, y=0.988)
fig.text(0.030, 0.940,
         f"Both axes carry {A.q}, in the same unit: the pencil-to-pencil "
         f"scatter. The box distribution is therefore\n"
         f"narrow by a factor of {sd_ratio:.2f}, which is a real property of "
         f"the box and not a choice of normalisation. The dotted\n"
         f"line is where a box would sit if it inherited the whole of its "
         f"pencil's displacement.\n\n"
         f"The selected pencils move far to the right and hardly at all "
         f"upward: the criterion picks an unusual\n"
         f"region out of an ordinary box.",
         fontsize=9.0, va="top", ha="left", color="0.15", linespacing=1.5)
fig.savefig(A.out)
print(f"wrote {A.out}\n")
print(f"{A.q}:  sd_box/sd_pencil = {sd_ratio:.4f}   rho = {rho:+.4f}")
hdr = f"{'selection':<26}{'n':>8}{'pencil':>10}{'box':>10}{'inherited':>11}"
print(hdr); print("-"*len(hdr))
for idx, _, _, _, lab in SELECTIONS:
    dx, dy = x[idx].mean(), y[idx].mean()
    print(f"{lab.split('  ')[0]:<26}{len(idx):>8,}{dx:>+10.3f}{dy:>+10.4f}"
          f"{dy/dx if abs(dx) > 1e-3 else np.nan:>10.1%}")
