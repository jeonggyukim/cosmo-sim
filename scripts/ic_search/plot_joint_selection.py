#!/usr/bin/env python3
"""The conditioning identity as a picture: two quantities, jointly.

The summary figure reports one number per property, the shift of its mean. That
number is the end of an argument whose middle is a joint distribution, and this
figure draws that middle.

Both axes are in units of each quantity's own scatter, z = (Q - mean)/sd. In
those units the Gaussian conditional mean of Eq. (8) is a straight line through
the origin whose slope is the correlation coefficient itself, so the prediction
can be drawn on the plot rather than quoted beside it.

The criterion selects on the horizontal axis alone. What should be visible: the
kept subvolumes form a vertical band displaced to the right, and that band sits
higher on the vertical axis than the parent by the amount the line predicts, even
though nothing in the criterion mentions the vertical quantity. The tighter cut
lands almost on top of the looser one, which is the saturation.

Usage:
    python plot_joint_selection.py --data DIR [--y "tidal shear R=9"]
                                   [--keep 0.01] [--keep2 0.001] [--out PNG]
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
ap.add_argument("--x", default="large-scale power",
                help="the quantity the criterion acts on")
ap.add_argument("--y", default="tidal shear R=9",
                help="a quantity the criterion never measures")
ap.add_argument("--keep", type=float, default=0.01)
ap.add_argument("--keep2", type=float, default=0.001)
ap.add_argument("--nbin", type=int, default=160)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "joint_selection.png"))
A = ap.parse_args()

crit_th, crit_wn, C, meta, _ = chunkio.load(A.data, A.species)
nseed, npen = crit_th.shape
crit_th, crit_wn = crit_th.ravel(), crit_wn.ravel()
C = {n: v.ravel() for n, v in C.items()}
for name in (A.x, A.y):
    if name not in C:
        raise SystemExit(f"no quantity named {name!r}. Available: "
                         + ", ".join(sorted(C)))

nk = max(1, int(round(A.keep*len(crit_th))))
nk2 = max(1, int(round(A.keep2*len(crit_th))))
order = np.argsort(crit_th)

# Three selections. The first two ask the pencil to match the raw linear theory,
# at two cuts a factor of ten apart. The third asks it to match the theory
# convolved with the pencil window, the curve the estimator is actually unbiased
# for, and is the control: that target demands nothing atypical, so its cloud
# should sit over the middle of the parent distribution on both axes. Drawn
# first, so the selections the figure is about are not painted over by it.
#
# Colour follows the rest of the figures in this directory: red and orange for
# the proposed criterion, green for the window-matched control. Blue is not free
# here, since region_vs_box uses it for the whole box.
SELECTIONS = [
    (np.argsort(crit_wn)[:nk], "C2", 3.0, 0.30,
     f"match theory $\\ast$ window, keeping {100*A.keep:g}%  (control)"),
    (order[:nk], "C3", 3.0, 0.30,
     f"match raw theory, keeping {100*A.keep:g}%"),
    (order[:nk2], "C1", 11.0, 0.85,
     f"match raw theory, keeping {100*A.keep2:g}%"),
]


def z(name):
    v = C[name]
    return (v - v.mean())/v.std()


X, Y = z(A.x), z(A.y)
rho = float(np.corrcoef(X, Y)[0, 1])

# ---- what each axis actually is ----
# A reader cannot judge a correlation between two quantities without knowing how
# each was reduced to a number. These are the definitions of section 4.1 of
# ic_search.tex, written out so the figure stands on its own.
import re

_dkp, _kny = float(meta["dkperp"]), float(meta["kny"])
WEB = {"knot": "three", "filament": "two", "sheet": "one", "void": "no"}


def describe(name):
    m = re.search(r"R=(\d+(?:\.\d+)?)$", name)
    R = m.group(1) if m else None
    sm = (f"$\\delta_R$ is the density contrast smoothed with a Gaussian of "
          f"radius $R = {R}$ Mpc/$h$")
    if name == "large-scale power":
        return (r"$\left\langle \hat P(k) \,/\, [P_{\rm th} \ast |\widetilde W|^2](k)"
                r"\right\rangle$ averaged over "
                f"$k \\leq 2\\Delta k_\\perp = {2*_dkp:.3f}$ " + r"$h\,$Mpc$^{-1}$,"
                "\n     the band the criterion uses. Divided by the convolved "
                "theory, so $1$ means an ordinary region, not a matching one.")
    if name == "small-scale power":
        return (r"the same ratio $\hat P / [P_{\rm th} \ast |\widetilde W|^2]$ "
                f"averaged over ${2*_dkp:.3f} < k \\leq 0.9\\,k_{{\\rm Ny}} = "
                f"{0.9*_kny:.2f}$ " + r"$h\,$Mpc$^{-1}$," "\n     that is, over the "
                "scales the criterion never looks at.")
    if name.startswith("tidal shear"):
        return (r"$\left\langle s_{ij}s_{ij}\right\rangle_V^{1/2}$ with "
                r"$s_{ij} = T_{ij} - \frac{1}{3}\delta_R\delta_{ij}$ and "
                r"$T_{ij}(\mathbf{k}) = k_ik_j\,\delta_R(\mathbf{k})/k^2$;"
                "\n     " + sm + ", and $\\langle\\cdot\\rangle_V$ averages over "
                "the cells of the subvolume.")
    for w, n in WEB.items():
        if name.startswith(f"{w} fraction"):
            return (f"fraction of the subvolume's cells with {n} positive "
                    r"eigenvalues of $T_{ij}$, a positive"
                    "\n     eigenvalue meaning collapse along that axis; "
                    + sm + ".")
    if name.startswith("mean overdensity"):
        return r"$\left\langle \delta_R \right\rangle_V$, the smoothed density " \
               "contrast averaged over the subvolume;\n     " + sm + "."
    if name.startswith("ellipticity"):
        return (r"$(\lambda_1-\lambda_3)/2|\lambda|$ from the eigenvalues of "
                r"$T_{ij}$, normalised by $|\lambda| = (\sum_i\lambda_i^2)^{1/2}$"
                "\n     rather than by the trace, which passes through zero; "
                + sm + ".")
    if name == "bulk flow":
        return (r"$|\left\langle \mathbf{\Psi} \right\rangle_V|$, the "
                r"Zel'dovich displacement $\mathbf{\Psi}(\mathbf{k}) = "
                r"i\mathbf{k}\,\delta(\mathbf{k})/k^2$ averaged over the"
                "\n     subvolume as a vector and then taken in magnitude, so it "
                "depends on the phases of the modes.")
    if name.startswith("env contrast"):
        return ("the subvolume's mean density minus that of the eight "
                "subvolumes surrounding it,\n     which asks about its "
                "neighbourhood rather than about the box; " + sm + ".")
    return "see section 4.1 of ic\\_search.tex"

# ---- layout: joint panel with a marginal histogram on each axis ----
fig = plt.figure(figsize=(9.6, 9.9))
gs = fig.add_gridspec(2, 2, width_ratios=(4.4, 1.15), height_ratios=(1.15, 4.4),
                      wspace=0.045, hspace=0.045,
                      left=0.088, right=0.978, bottom=0.132, top=0.775)
ax = fig.add_subplot(gs[1, 0])
axx = fig.add_subplot(gs[0, 0], sharex=ax)
axy = fig.add_subplot(gs[1, 1], sharey=ax)

# The view has to hold the parent population and every selected point, so the
# limits come from both. Percentiles alone would clip the selection, which is
# the part of the plane the figure exists to show.
lo_x, hi_x = np.percentile(X, [0.02, 99.98])
lo_y, hi_y = np.percentile(Y, [0.02, 99.98])
for idx, *_ in SELECTIONS:
    lo_x, hi_x = min(lo_x, X[idx].min()), max(hi_x, X[idx].max())
    lo_y, hi_y = min(lo_y, Y[idx].min()), max(hi_y, Y[idx].max())
pad_x, pad_y = 0.04*(hi_x - lo_x), 0.04*(hi_y - lo_y)
XL, YL = (lo_x - pad_x, hi_x + pad_x), (lo_y - pad_y, hi_y + pad_y)

# ---- parent population as filled contours ----
H, xe, ye = np.histogram2d(X, Y, bins=A.nbin, range=(XL, YL))
# The outermost contour traces cells holding a handful of points each, so
# unsmoothed it breaks into speckle that reads as structure. A Gaussian of about
# one bin removes that without moving the enclosed fractions measurably.
from scipy.ndimage import gaussian_filter
H = gaussian_filter(H, 1.2)
# Contour levels enclosing a stated fraction of the points, which is what a
# reader assumes a contour means. Sorting the cells and walking down the
# cumulative count gives the density threshold for each fraction.
flat = np.sort(H.ravel())[::-1]
cum = np.cumsum(flat)/H.sum()
levels = [flat[np.searchsorted(cum, f)] for f in (0.99, 0.9, 0.68, 0.38)]
xc, yc = 0.5*(xe[1:] + xe[:-1]), 0.5*(ye[1:] + ye[:-1])
ax.contourf(xc, yc, H.T, levels=levels + [H.max()],
            colors=["0.90", "0.80", "0.68", "0.55"], zorder=0)
ax.contour(xc, yc, H.T, levels=levels, colors="0.45", linewidths=0.5, zorder=1)

for j, (idx, col, size, al, lab) in enumerate(SELECTIONS):
    ax.scatter(X[idx], Y[idx], s=size, marker="o", c=col, alpha=al,
               linewidths=0, zorder=3 + j, label=lab)

# ---- the prediction: in these units the conditional mean has slope rho ----
xs = np.array(XL)
ax.plot(xs, rho*xs, color="k", lw=1.6, ls="--", zorder=4,
        label=f"$\\langle y\\,|\\,x\\rangle = \\rho\\,x$,  $\\rho = {rho:+.3f}$")
ax.axhline(0, color="0.35", lw=0.9, zorder=2)
ax.axvline(0, color="0.35", lw=0.9, zorder=2)

# Where each selection actually landed, against where the line says it should.
for idx, col, _, _, _ in SELECTIONS:
    dx, dy = X[idx].mean(), Y[idx].mean()
    ax.plot([dx], [dy], marker="P", ms=13, mfc=col, mec="k", mew=1.2, zorder=8)
    ax.plot([dx], [rho*dx], marker="_", ms=17, mec="k", mew=2.4, zorder=7)

ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel(f"{A.x}   [standard deviations]")
ax.set_ylabel(f"{A.y}   [standard deviations]")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

# ---- marginals ----
bx = np.linspace(*XL, 70)
by = np.linspace(*YL, 70)
axx.hist(X, bins=bx, color="0.78", edgecolor="none", density=True)
axy.hist(Y, bins=by, color="0.78", edgecolor="none", density=True,
         orientation="horizontal")
for idx, col, _, _, _ in SELECTIONS:
    # The tighter cut holds a tenth as many points, so it gets half the bins;
    # a density histogram stays comparable across different bin widths.
    step = 1 if len(idx) == nk else 2
    axx.hist(X[idx], bins=bx[::step], histtype="step", lw=1.6, color=col,
             density=True)
    axy.hist(Y[idx], bins=by[::step], histtype="step", lw=1.6, color=col,
             density=True, orientation="horizontal")
    axx.axvline(X[idx].mean(), color=col, lw=1.1, ls="--")
    axy.axhline(Y[idx].mean(), color=col, lw=1.1, ls="--")
axx.axvline(0, color="0.35", lw=0.9)
axy.axhline(0, color="0.35", lw=0.9)
for a in (axx, axy):
    a.set_yticks([]) if a is axx else a.set_xticks([])
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
plt.setp(axx.get_xticklabels(), visible=False)
plt.setp(axy.get_yticklabels(), visible=False)
axx.spines["left"].set_visible(False)
axy.spines["bottom"].set_visible(False)
axx.set_ylabel("")
axy.set_xlabel("")

# ---- the numbers the figure is evidence for ----
# Inside the joint panel, in its empty lower-right corner. Put above the axes
# instead and it lands on the marginal histogram, which shares that strip.
SHORT = {"C2": "control", "C3": f"raw, {100*A.keep:g}%",
         "C1": f"raw, {100*A.keep2:g}%"}
rows = [f"$\\rho = {rho:+.3f}$"]
for idx, col, _, _, _ in SELECTIONS:
    dx, dy = X[idx].mean(), Y[idx].mean()
    rows.append(f"{SHORT[col]}:   $\\Delta_x = {dx:+.2f}$,   "
                f"$\\rho\\Delta_x = {rho*dx:+.2f}$,   "
                f"measured $\\Delta_y = {dy:+.2f}$")
ax.text(0.985, 0.022, "\n".join(rows), transform=ax.transAxes, fontsize=8.8,
        va="bottom", ha="right", color="0.15", linespacing=1.5, zorder=7,
        bbox=dict(fc="white", ec="0.75", alpha=0.94, pad=4.5))

fig.suptitle(f"{nseed:,} realizations $\\times$ {npen} subvolumes "
             f"$= {len(X):,}$ measurements, $N={int(meta['N'])}^3$, "
             f"$L={meta['L']:g}$ Mpc/$h$, 2LPT, $\\delta(q)$ matter,\n"
             f"pencil $=(L/8)^2 \\times L$",
             fontsize=11, y=0.988)

# ---- what the two axes are, above the panels rather than inside a caption ----
fig.text(0.030, 0.938,
         f"horizontal, $x$ = {A.x}:  " + describe(A.x) + "\n\n"
         f"vertical, $y$ = {A.y}:  " + describe(A.y) + "\n\n"
         r"Both axes are then standardised, $z = (Q - \langle Q\rangle)/\sigma_Q$"
         r", over all subvolumes, so a value of $1$ is one standard deviation of"
         "\n     the region-to-region scatter. In these units the Gaussian "
         r"conditional mean $\langle y\,|\,x\rangle = \rho\,(\sigma_y/\sigma_x)"
         r"(x-\bar x)$ is a line through the origin"
         "\n     of slope $\\rho$ exactly, which is why the prediction can be "
         "drawn on the plot.",
         fontsize=8.6, va="top", ha="left", color="0.15", linespacing=1.45)
fig.text(0.5, 0.016,
         f"One point per pencil subvolume, not per box. Filled contours enclose "
         f"38, 68, 90 and 99 per cent of all {len(X):,} of them.\n"
         "Crosses mark the mean of each kept sample; the short black bar is "
         "where the dashed line predicts that mean.",
         fontsize=8.5, color="0.35", va="bottom", ha="center", linespacing=1.5)
fig.savefig(A.out)
print(f"wrote {A.out}\n")

print(f"{A.x}  vs  {A.y}")
print(f"rho = {rho:+.4f}   over {len(X):,} subvolumes "
      f"({nseed:,} realizations x {npen})")
hdr = f"{'selection':>16}{'n':>9}{'dx':>9}{'rho*dx':>10}{'dy':>9}{'dy/dx':>9}"
print(hdr); print("-"*len(hdr))
for idx, col, _, _, _ in SELECTIONS:
    dx, dy = X[idx].mean(), Y[idx].mean()
    print(f"{SHORT[col]:>16}{len(idx):>9,}{dx:>+9.3f}{rho*dx:>+10.3f}"
          f"{dy:>+9.3f}{dy/dx if abs(dx) > 1e-6 else np.nan:>9.3f}")
