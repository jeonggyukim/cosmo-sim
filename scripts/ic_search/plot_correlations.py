#!/usr/bin/env python3
"""How many independent things the summary figure is actually showing.

Twelve rows of shifts read as twelve pieces of evidence. They are not. The four
web fractions add up to one at every cell, so only three of them can be
independent, and two smoothing radii are the same field seen at two scales.

Left panel is the correlation between every pair of measured quantities. Right
panel is the one column that explains the whole result: how strongly each
quantity tracks the large-scale power that the selection acts on. A quantity's
shift under selection is that number times the shift in the power itself, so
the right panel predicts the summary figure.

Usage:
    python plot_correlations.py --data DIR [--out PNG]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import paths, chunkio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"))
ap.add_argument("--nchunk", type=int, default=120)
ap.add_argument("--out", default=os.path.join(paths.FIGS, "correlations.png"))
A = ap.parse_args()

_, _, C, meta, _ = chunkio.load(A.data, "matter", nchunk=A.nchunk)
nseed, npen = C["large-scale power"].shape
C = {n: v.ravel() for n, v in C.items()}

ORDER = ["large-scale power", "small-scale power"] + \
        [n for n in C if n.startswith("tidal shear")] + \
        [n for n in C if n.startswith("mean overdensity")] + \
        [n for n in C if n.split()[0] in ("knot", "filament", "sheet", "void")] + \
        [n for n in C if n == "bulk flow"]
ORDER = [n for n in ORDER if n in C]
# A quantity that is undefined, such as the interior of a region smoothed on half
# its own width, or one with no scatter, has no correlation with anything. Left
# in, either makes the matrix impossible to diagonalise.
usable = [n for n in ORDER
          if np.all(np.isfinite(C[n])) and np.std(C[n]) > 0]
dropped = [n for n in ORDER if n not in usable]
if dropped:
    print("not correlated, undefined or constant: " + ", ".join(dropped) + "\n")
ORDER = usable
X = np.stack([C[n] for n in ORDER])
R = np.corrcoef(X)

short = [n.replace("fraction ", "").replace("tidal shear", "shear")
          .replace("mean overdensity", "density").replace("large-scale", "large")
          .replace("small-scale", "small").replace("R=", "R") for n in ORDER]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.2, 6.4),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

im = a1.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
a1.set_xticks(range(len(short))); a1.set_xticklabels(short, rotation=90, fontsize=7.5)
a1.set_yticks(range(len(short))); a1.set_yticklabels(short, fontsize=7.5)
for i in range(len(short)):
    for j in range(len(short)):
        if abs(R[i, j]) > 0.45:
            a1.text(j, i, f"{R[i,j]:+.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=5.6, color="white" if abs(R[i, j]) > 0.75 else "0.2")
fig.colorbar(im, ax=a1, fraction=0.046, shrink=0.85, label="correlation")
w = np.linalg.eigvalsh(R)[::-1]
a1.set_title(f"Most of these measure the same few things\n"
             f"{(w > 1).sum()} of {len(w)} directions carry more than their share; "
             f"two are exactly zero,\nbecause the four web fractions must add to one",
             fontsize=9.5)

i0 = ORDER.index("large-scale power")
rho = R[i0]
y = np.arange(len(ORDER))[::-1]
col = ["C3" if r > 0.05 else "C0" if r < -0.05 else "0.6" for r in rho]
a2.barh(y, rho, color=col, height=0.68)
a2.axvline(0, color="0.3", lw=1.0)
a2.set_yticks(y); a2.set_yticklabels(short, fontsize=8)
a2.set_xlabel("correlation with the large-scale power the selection acts on")
a2.set_xlim(-0.55, 1.08)
for yy, r in zip(y, rho):
    a2.text(r + (0.02 if r >= 0 else -0.02), yy, f"{r:+.2f}",
            va="center", ha="left" if r >= 0 else "right", fontsize=7.5)
a2.set_title("This column predicts the whole result\n"
             "a quantity shifts by this number times the shift in the power",
             fontsize=9.5)

fig.suptitle(f"What the selected quantities have in common: "
             f"{nseed:,} realizations x {npen} pencils, "
             f"$N=128^3$, $L=700$ Mpc/$h$", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(A.out)
print(f"wrote {A.out}\n")
print(f"eigenvalues: {np.round(w, 2)}")
print(f"effective independent directions (participation ratio): "
      f"{w.sum()**2/(w**2).sum():.2f} of {len(w)}")
for n, r in sorted(zip(ORDER, rho), key=lambda t: -abs(t[1])):
    print(f"  {n:<26} {r:+.3f}")
