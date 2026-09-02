#!/usr/bin/env python3
"""How atypical the selected region is, as a function of how many seeds were searched.

The method does not keep the best 5% of candidates. It keeps the best one out of
however many seeds were tried, so the quantity that matters is what "the best of
N" looks like, not what a fixed percentile looks like.

Measuring that needs the experiment repeated. The realizations are partitioned
into groups of N, the best pencil in each group is kept, and the spread over
groups gives the uncertainty. A sample of M realizations therefore supports
searches of size N with M/N independent repeats, and the error bar grows as N
does. Where it grows past the effect, the measurement has stopped saying
anything, and the figure marks that point rather than extrapolating through it.

Usage:
    python plot_search_size.py --data DIR [--out PNG]
"""
import argparse, glob, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"))
ap.add_argument("--species", default="matter")
ap.add_argument("--out", default=os.path.join(paths.FIGS, "search_size.png"))
ap.add_argument("--min-repeats", type=int, default=8,
                help="smallest number of independent repeats worth plotting")
A = ap.parse_args()


def load(path, species):
    chunks = sorted(glob.glob(f"{path}/chunk_*.hdf5"))
    if not chunks:
        raise SystemExit(f"no chunk_*.hdf5 under {path}")
    with h5py.File(chunks[0]) as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
        SP = names.index(species)
        k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
        dkperp = f.attrs["dkperp"]
        RS = f["smooth_R"][:] if "smooth_R" in f else []
    lo = k <= 2*dkperp

    crit, cols, box = [], {}, {}
    for fn in chunks:
        with h5py.File(fn) as f:
            P = f["P_pencil"][:, SP]                      # (nseed, npencil, nk)
            crit.append(np.sqrt((np.log(P[:, :, lo]/P_th[lo])**2).mean(2)))
            cols.setdefault("large-scale power", []).append((P[:, :, lo]/P_win[lo]).mean(2))
            for r, R in enumerate(RS):
                if "shear" in f:
                    cols.setdefault(f"tidal shear R={R:.0f}", []).append(f["shear"][:, r])
                    cols.setdefault(f"mean overdensity R={R:.0f}", []).append(f["dbar"][:, r])
                if "webtype" in f:
                    for w, wn in ((0, "knot"), (3, "void")):
                        cols.setdefault(f"{wn} fraction R={R:.0f}", []).append(
                            f["webtype"][:, r, :, w])
                if "shear_box" in f:
                    box.setdefault(f"tidal shear R={R:.0f}", []).append(f["shear_box"][:, r])
            if "bulk" in f:
                cols.setdefault("bulk flow", []).append(np.linalg.norm(f["bulk"][:], axis=-1))
    crit = np.concatenate(crit)
    cols = {n: np.concatenate(v) for n, v in cols.items()}
    box = {n: np.concatenate(v) for n, v in box.items()}
    return crit, cols, box


crit, cols, box = load(A.data, A.species)
nseed, npen = crit.shape
print(f"{nseed:,} realizations x {npen} pencils")

# One pencil per realization makes the draws independent: pencils inside a box
# share its modes, so a group of N boxes must contribute N candidates, not 24N.
rng = np.random.default_rng(0)
pick = rng.integers(0, npen, nseed)
c1 = crit[np.arange(nseed), pick]

# The radii differ between runs: the first sweeps fixed them at 20 and 40 Mpc/h,
# later ones set them as fractions of the pencil width, giving 9, 22 and 44.
PREFER = ["tidal shear", "large-scale power", "knot fraction", "void fraction",
          "bulk flow"]
SHOW = [n for stem in PREFER for n in cols if n.startswith(stem)]
SHOW = [n for n in SHOW if not n.startswith("mean overdensity")][:8]

SIZES = [n for n in (3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)
         if nseed//n >= A.min_repeats]

res = {}
for name in SHOW:
    T1 = cols[name][np.arange(nseed), pick]
    mu, sd = T1.mean(), T1.std()
    pts = []
    for N in SIZES:
        ngroup = nseed//N
        c = c1[:ngroup*N].reshape(ngroup, N)
        t = T1[:ngroup*N].reshape(ngroup, N)
        win = t[np.arange(ngroup), c.argmin(1)]        # best-matching seed per group
        pts.append(((win.mean() - mu)/sd, win.std()/sd/np.sqrt(ngroup), ngroup))
    res[name] = np.array(pts)

fig, ax = plt.subplots(figsize=(8.6, 5.8))
for i, name in enumerate(SHOW):
    v, e, ng = res[name].T
    ax.errorbar(SIZES, v, yerr=e, marker="o", ms=4.5, lw=1.5, capsize=3,
                label=name, color=f"C{i}")
ax.axhline(0.0, color="0.35", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel("number of seeds searched, keeping the best one")
ax.set_ylabel("shift of the selected region  [standard deviations]")
ax.legend(fontsize=8.5, framealpha=0.95, loc="upper left")
ax.grid(alpha=0.25)
ax.set_title(f"What the search actually produces, as a function of its size\n"
             f"{nseed:,} realizations; the error bar is the spread over "
             f"{nseed:,}/N independent repeats", fontsize=10.5)
fig.tight_layout()
fig.savefig(A.out)
print(f"wrote {A.out}\n")

hdr = f"{'quantity':<24}" + "".join(f"{N:>12}" for N in SIZES)
print(hdr); print("-"*len(hdr))
for name in SHOW:
    v, e, ng = res[name].T
    print(f"{name:<24}" + "".join(f"{a:+7.3f}±{b:.3f}" for a, b in zip(v, e)))
print(f"\nrepeats per size: " + ", ".join(f"N={N}: {nseed//N}" for N in SIZES))
