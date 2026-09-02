#!/usr/bin/env python3
"""How the bias depends on the scale it is measured on.

A criterion acting on the region's large-scale power need not distort every
scale equally. Measuring each quantity on several smoothing radii says which
scales it reaches, and the radius can be read as a mass: a sphere of radius R at
the mean matter density encloses (4/3) pi R^3 rho_m, so 9 Mpc/h is a rich cluster
and 44 Mpc/h is larger than any collapsed object.

The interior-only variants are measured on the region trimmed by one smoothing
radius from each long face, so no cell entering the average was smoothed with
material from outside the region. Where the two agree, the number describes the
region; where they part, it describes the region together with its surroundings.

Usage:
    python plot_scale_dependence.py --data DIR [--keep 0.01] [--out PNG]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import paths, chunkio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128b"))
ap.add_argument("--keep", type=float, default=0.01)
ap.add_argument("--nboot", type=int, default=200)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "scale_dependence.png"))
A = ap.parse_args()

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13.5,
                     "legend.fontsize": 12, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5})

crit_raw, crit_win, COLS, meta, _ = chunkio.load(A.data, "matter")
nseed, npen = crit_raw.shape
shift, shift_err, radii_for = chunkio.selection_stats(
    {"raw": crit_raw, "win": crit_win}, COLS, A.keep)
N, L = meta["N"], meta["L"]
lperp = L/8.0

# Its own palette, distinct from the run-comparison figure, where colour means
# which curve a region was matched against rather than which quantity it is.
SERIES = [("tidal shear",          "tab:purple", "-",  "o", "tidal shear"),
          ("tidal shear interior", "tab:purple", "--", "s", "tidal shear, interior only"),
          ("knot fraction",        "tab:orange", "-",  "^", "knot fraction"),
          ("void fraction",        "tab:brown",  "-",  "v", "void fraction"),
          ("mean overdensity",     "0.5",        "-",  "D", "mean overdensity")]

fig, ax = plt.subplots(figsize=(9.6, 7.2))

for stem, col, ls, mk, lab in SERIES:
    rs = radii_for(stem)
    if not rs:
        continue
    x = [r for r, _ in rs]
    y = [shift(n, "raw") for _, n in rs]
    e = [shift_err(n, "raw", A.nboot) for _, n in rs]
    if not np.any(np.isfinite(y)):
        continue
    ax.errorbar(x, y, yerr=e, color=col, ls=ls, marker=mk, ms=7, lw=1.8,
                capsize=3.5, label=lab)

ax.axhline(0.0, color="0.3", lw=1.0)
ax.axvspan(6.0, 12.0, color="tab:blue", alpha=0.07, zorder=0)
ax.text(8.5, 0.03, "galaxy clusters\n($\\sim 10^{14}\\,h^{-1}M_\\odot$)",
        transform=ax.get_xaxis_transform(), fontsize=11.5, color="tab:blue",
        ha="center", va="bottom")
ax.axvline(lperp, color="0.55", lw=1.2, ls=":")
ax.text(lperp*0.93, 0.03, "region width ", transform=ax.get_xaxis_transform(),
        fontsize=11.5, color="0.4", ha="right", va="bottom")

ax.set_xscale("log")
ax.set_xlabel("smoothing radius  [Mpc/$h$]")
ax.set_ylabel("shift of the pencil zoom-in region  [standard deviations]")

top = ax.secondary_xaxis(
    "top", functions=(lambda r: chunkio.MASS_PER_R3*np.maximum(r, 1e-6)**3,
                      lambda m: (np.maximum(m, 1e-6)/chunkio.MASS_PER_R3)**(1/3)))
top.set_xlabel("mass enclosed at the mean density  [$h^{-1} M_\\odot$]", labelpad=8)

ax.legend(framealpha=0.95, loc="upper right")
ax.grid(alpha=0.25)
ax.set_title(f"The distortion is largest on cluster scales\n"
             f"{nseed:,} realizations $\\times$ {npen} subvolumes, "
             f"$N={int(N)}^3$, $L={L:g}$ Mpc/$h$, keeping the closest "
             f"{100*A.keep:g}%", pad=42)

fig.tight_layout()
fig.savefig(A.out)
print(f"wrote {A.out}\n")
for stem, _, _, _, _ in SERIES:
    for r, n in radii_for(stem):
        m = chunkio.MASS_PER_R3*r**3
        print(f"  {n:<34} {shift(n, 'raw'):+7.3f} +/- "
              f"{shift_err(n, 'raw', A.nboot):.3f}   (M = {m:.1e})")
