#!/usr/bin/env python3
"""Three cases side by side: two ways of choosing a region, and the box it sits in.

  - a pencil chosen because its P(k) best matches the raw linear theory,
  - a pencil chosen because its P(k) best matches the theory convolved with the
    pencil window, which is the curve the estimator is unbiased for,
  - the whole box that contains it.

The second is the control: matching the convolved curve asks for nothing
unusual, so whatever separates it from the first is the effect of the target
curve rather than of selecting. The third asks how far the effect reaches: the
pencil is a sixty-fourth of the box by volume but shares its largest modes, so
selecting the pencil conditions the box a little. If the box moves much less than
the region, what a resimulation inherits is an unusual region in an ordinary
universe.

Usage:
    python plot_region_vs_box.py --data DIR [--keep 0.01] [--out PNG]
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
ap.add_argument("--out", default=os.path.join(paths.FIGS, "region_vs_box.png"))
A = ap.parse_args()

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13.5,
                     "legend.fontsize": 12, "xtick.labelsize": 12.5,
                     "ytick.labelsize": 12.5})

crit_raw, crit_win, COLS, meta, _ = chunkio.load(A.data, "matter")
nseed, npen = crit_raw.shape
shift, shift_err, radii_for = chunkio.selection_stats(
    {"raw": crit_raw, "win": crit_win}, COLS, A.keep)
N, L = meta["N"], meta["L"]

rows = []
for stem in ("tidal shear", "knot fraction", "void fraction", "mean overdensity"):
    for r, name in radii_for(stem):
        box = f"{stem} box R={r:.0f}"
        if box in COLS:
            rows.append((f"{stem}, $R$ = {r:g}", name, box))

if not rows:
    raise SystemExit(f"{A.data} records no whole-box quantities to compare against")

y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(11.0, 0.52*len(rows) + 3.6))

# Colour means which case, not which quantity: red for the proposed criterion,
# green for the control, blue for the box.
CASES = [(0.24, "C3", "D", "raw", 1, "pencil matched to the raw theory"),
         (0.00, "C2", "s", "win", 1, "pencil matched to theory $\\ast$ window"),
         (-0.24, "C0", "o", "raw", 2, "whole box")]
for off, col, mk, which, col_idx, lab in CASES:
    names = [row[col_idx] for row in rows]
    v = [shift(n, which) for n in names]
    e = [shift_err(n, which, A.nboot) for n in names]
    ax.errorbar(v, y + off, xerr=e, fmt=mk, color=col, ms=7, lw=1.6,
                capsize=3.5, label=lab)

ax.axvline(0.0, color="0.3", lw=1.2)
for yy in y[:-1]:
    ax.axhline(yy - 0.5, color="0.9", lw=0.8, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
ax.set_xlabel("shift  [standard deviations]")
ax.legend(framealpha=0.95, loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.set_title(f"Two ways of choosing a region, and the box that contains it\n"
             f"{nseed:,} realizations $\\times$ {npen} subvolumes\n"
             f"$N={int(N)}^3$, $L={L:g}$ Mpc/$h$, keeping the closest "
             f"{100*A.keep:g}%", pad=14, fontsize=13.5)

fig.tight_layout()
fig.savefig(A.out)
print(f"wrote {A.out}\n")
print(f"{'quantity':<26}{'raw':>9}{'window':>9}{'box':>9}")
for lab, name, box in rows:
    print(f"{lab.replace('$', ''):<26}{shift(name, 'raw'):+9.3f}"
          f"{shift(name, 'win'):+9.3f}{shift(box, 'raw'):+9.3f}")
