#!/usr/bin/env python3
"""How the bias depends on scale, and whether the parent box is affected at all.

Two questions the earlier runs could not answer, both now measurable because the
sweep records the smoothing radii as fractions of the region width and measures
the whole box exactly rather than estimating it from the subvolumes.

Left: the shift in each quantity against the smoothing radius it was measured on.
A criterion that acts on the region's large-scale power need not bias every scale
equally, and the radius dependence says which scales it reaches. The interior
variants, measured on the region trimmed by a margin so that no cell was smoothed
with material from outside it, are drawn alongside: where they agree, the number
describes the region; where they part, it describes the region and its
surroundings together.

Right: the same shift for the region and for the whole box that contains it. The
box is the universe the region was drawn from. If the region moves and the box
does not, the search produced an unusual region inside an ordinary universe, and
what a resimulation inherits is the region.

Usage:
    python plot_scale_and_box.py --data DIR [--keep 0.01] [--out PNG]
"""
import argparse, os, re
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
ap.add_argument("--out", default=os.path.join(paths.FIGS, "scale_and_box.png"))
A = ap.parse_args()

crit, _, COLS, meta, _ = chunkio.load(A.data, "matter")
nseed, npen = crit.shape
crit_f = crit.ravel()
COLS = {n: v.ravel() for n, v in COLS.items()}
N, L = meta["N"], meta["L"]
lperp = L/8.0

rng = np.random.default_rng(0)
nkeep = max(1, int(round(A.keep*len(crit_f))))


def shift(name, idx=None):
    """Shift of the kept subvolumes in units of the population scatter."""
    T = COLS[name]
    c = crit_f
    if idx is not None:
        T, c = T[idx], c[idx]
    n = max(1, int(round(A.keep*len(c))))
    keep = np.argpartition(c, n)[:n]
    sd = T.std()
    return (T[keep].mean() - T.mean())/sd if sd > 0 else np.nan


def shift_err(name):
    """Bootstrap over realizations, which is the independent unit."""
    boot = np.empty(A.nboot)
    for b in range(A.nboot):
        g = rng.integers(0, nseed, nseed)
        boot[b] = shift(name, (g[:, None]*npen + np.arange(npen)).ravel())
    return float(np.std(boot))


def radii_for(stem):
    """The smoothing radii present for a given quantity, sorted."""
    out = []
    for n in COLS:
        m = re.fullmatch(rf"{stem} R=(\d+(?:\.\d+)?)", n)
        if m:
            out.append((float(m.group(1)), n))
    return sorted(out)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.6))

# ---- left: dependence on the smoothing radius -------------------------------
SERIES = [("tidal shear", "C0", "-", "o", "tidal shear"),
          ("tidal shear interior", "C0", "--", "s", "tidal shear, interior only"),
          ("knot fraction", "C3", "-", "o", "knot fraction"),
          ("void fraction", "C2", "-", "o", "void fraction"),
          ("mean overdensity", "0.45", "-", "o", "mean overdensity")]
plotted = 0
for stem, col, ls, mk, lab in SERIES:
    rs = radii_for(stem)
    if not rs:
        continue
    x = [r for r, _ in rs]
    y = [shift(n) for _, n in rs]
    e = [shift_err(n) for _, n in rs]
    if not np.any(np.isfinite(y)):
        continue
    a1.errorbar(x, y, yerr=e, color=col, ls=ls, marker=mk, ms=5, lw=1.6,
                capsize=3, label=lab)
    plotted += 1
a1.axhline(0.0, color="0.3", lw=1.0)
a1.axvline(lperp, color="0.55", lw=1.0, ls=":")
a1.text(lperp, 0.02, " region width", transform=a1.get_xaxis_transform(),
        fontsize=9, color="0.4")
a1.set_xscale("log")
a1.set_xlabel("smoothing radius  [Mpc/$h$]")
a1.set_ylabel("shift of the pencil zoom-in region  [standard deviations]")
a1.set_title("Dependence on the smoothing radius", fontsize=11)
a1.legend(fontsize=9, framealpha=0.95)
a1.grid(alpha=0.25)

# ---- right: region against the box it came from ------------------------------
pairs = []
for stem in ("tidal shear", "knot fraction", "void fraction", "mean overdensity"):
    for r, name in radii_for(stem):
        boxname = f"{stem} box R={r:.0f}"
        if boxname in COLS:
            pairs.append((f"{stem}\nR = {r:g}", name, boxname))

if pairs:
    y = np.arange(len(pairs))[::-1]
    reg = [shift(n) for _, n, _ in pairs]
    box = [shift(b) for _, _, b in pairs]
    ereg = [shift_err(n) for _, n, _ in pairs]
    ebox = [shift_err(b) for _, _, b in pairs]
    a2.errorbar(reg, y + 0.13, xerr=ereg, fmt="D", color="C3", ms=5, lw=1.4,
                capsize=3, label="pencil zoom-in region")
    a2.errorbar(box, y - 0.13, xerr=ebox, fmt="o", color="C0", ms=5, lw=1.4,
                capsize=3, label="whole box")
    a2.axvline(0.0, color="0.3", lw=1.0)
    a2.set_yticks(y)
    a2.set_yticklabels([p[0].replace("\n", ", ") for p in pairs], fontsize=9)
    a2.set_xlabel("shift  [standard deviations]")
    a2.set_title("Pencil zoom-in region against the whole box", fontsize=11)
    a2.legend(fontsize=9, framealpha=0.95, loc="lower right")
    a2.grid(axis="x", alpha=0.25)
else:
    a2.text(0.5, 0.5, "this sweep records no whole-box quantities",
            ha="center", va="center", transform=a2.transAxes, color="0.4")
    a2.set_xticks([]); a2.set_yticks([])

fig.suptitle(f"Selecting on the subvolume spectrum: dependence on scale, and on "
             f"the box the region came from\n"
             f"{nseed:,} realizations x {npen} subvolumes, $N={int(N)}^3$, "
             f"$L={L:g}$ Mpc/$h$, keeping the closest {100*A.keep:g}%",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(A.out)
print(f"wrote {A.out}\n")

for stem, _, _, _, _ in SERIES:
    for r, n in radii_for(stem):
        print(f"  {n:<34} {shift(n):+7.3f} +/- {shift_err(n):.3f}")
