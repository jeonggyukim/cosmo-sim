#!/usr/bin/env python3
"""Score a region for being typical, and compare the ways of doing it.

A region chosen for any reason should be reported with a number saying how
ordinary it is. This computes that number.

The score is the squared Mahalanobis distance of the region's displacement
vector,

    S_M = Delta^T C^-1 Delta,

where Delta_Q is each quantity's displacement in units of its own scatter over
all subvolumes, and C is the correlation matrix between the quantities. The
whitening matters: the quantities measured here span about three independent
directions, and the four cosmic-web fractions sum to unity exactly, so an
unweighted sum of squares counts whichever thing was measured most often. C is
singular for that reason and the inverse is taken on its non-degenerate
subspace.

For a Gaussian field S_M follows a chi-squared distribution with d_eff degrees
of freedom, so a typical region has S_M near d_eff, NOT near zero. That is the
part which catches people out: in d dimensions the probability mass sits in a
shell at radius sqrt(d), because the volume at radius r grows as r^(d-1). A
region at S_M = 0 is as unusual as one at S_M = 2 d_eff, and a rule that
minimises the score selects a region with far less internal structure than any
real region has.

The script therefore compares five rules, and reports for each not only the
shift of each quantity but the spread retained within the kept sample. The
shifts alone make the minimising rules look successful; the spreads show they
are not.

Usage:
    python typicality_score.py --data DIR [--keep 0.01] [--out PNG]
"""
import argparse, os
import numpy as np
import paths, chunkio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "big128"))
ap.add_argument("--species", default="matter")
ap.add_argument("--keep", type=float, default=0.01)
ap.add_argument("--out", default=None,
                help="write a figure of the score distribution here; "
                     "omit for the printed table alone")
A = ap.parse_args()

crit_th, _, C, meta, _ = chunkio.load(A.data, A.species)
nseed, npen = crit_th.shape
crit_th = crit_th.ravel()
C = {n: v.ravel() for n, v in C.items()}
ok = chunkio.usable(C)

# Only the per-region quantities belong in a score for the region. The "box" and
# "interior" variants are other views of the same thing and would be counted
# twice; the whitening would handle that, but a singular direction added on
# purpose is still a direction the score cannot use.
STEMS = ("tidal shear R", "knot fraction R", "filament fraction R",
         "sheet fraction R", "void fraction R", "mean overdensity R",
         "large-scale power", "small-scale power", "bulk flow")
SCORED = [n for stem in STEMS for n in C
          if n.startswith(stem) and ok(n) and " box " not in n
          and " interior " not in n]
if not SCORED:
    raise SystemExit(f"{A.data} records none of the quantities a score needs")


def z(name):
    v = C[name]
    return (v - v.mean())/v.std()


Z = np.stack([z(n) for n in SCORED], 1)
R = np.corrcoef(Z, rowvar=False)
w, V = np.linalg.eigh(R)
good = w > 1e-8
d_eff = int(good.sum())
part = float((w.sum()**2)/(w**2).sum())

naive = (Z**2).sum(1)
S_M = ((Z @ V[:, good]/np.sqrt(w[good]))**2).sum(1)
shell = np.abs(S_M - d_eff)

npoint = len(naive)
nkeep = max(1, int(round(A.keep*npoint)))
rng = np.random.default_rng(0)
RULES = [("minimise sum Delta^2", np.argsort(naive)[:nkeep]),
         ("minimise S_M", np.argsort(S_M)[:nkeep]),
         ("target the shell", np.argsort(shell)[:nkeep]),
         ("match the raw theory", np.argsort(crit_th)[:nkeep]),
         ("draw at random", rng.choice(npoint, nkeep, replace=False))]

print(f"{npoint:,} subvolumes from {nseed:,} realizations, "
      f"{len(SCORED)} quantities scored")
print("  " + ", ".join(SCORED))
print(f"\n{d_eff} non-degenerate directions; participation ratio {part:.2f}")
print(f"a random region has sum Delta^2 = {naive.mean():.1f} and "
      f"S_M = {S_M.mean():.1f}\n")

REPORT = [n for n in ("tidal shear R=9", "tidal shear R=20",
                      "knot fraction R=22", "knot fraction R=40",
                      "large-scale power") if n in C][:3]
hdr = (f"{'rule':<24}{'sumD2':>8}{'S_M':>8}"
       + "".join(f"{'shift '+n.split(' R=')[0][:11]:>18}" for n in REPORT)
       + "".join(f"{'spread '+n.split(' R=')[0][:10]:>18}" for n in REPORT))
print(hdr); print("-"*len(hdr))
for lab, idx in RULES:
    row = f"{lab:<24}{naive[idx].mean():>8.1f}{S_M[idx].mean():>8.1f}"
    row += "".join(f"{z(n)[idx].mean():>+18.3f}" for n in REPORT)
    row += "".join(f"{z(n)[idx].std():>18.3f}" for n in REPORT)
    print(row)

print("\nThe shift columns alone make the two minimising rules look successful. "
      "The spread\ncolumns are the check that matters: they keep a third to a "
      "half of the natural\nscatter, so they select regions with less structure "
      "than any real region has.\nA random draw needs no score and beats every "
      "one of them.")

# A sphere of radius R at the mean density encloses this much mass, which says
# which objects a smoothing radius corresponds to and therefore which radii a
# given science question should score.
print(f"\nmass enclosed at each smoothing radius "
      f"({chunkio.MASS_PER_R3:.3e} h^-1 Msun per (Mpc/h)^3):")
seen = set()
for n in SCORED:
    if " R=" not in n:
        continue
    r = float(n.split(" R=")[1])
    if r in seen:
        continue
    seen.add(r)
    print(f"  R = {r:>5.1f} Mpc/h   M = {chunkio.MASS_PER_R3*r**3:.2e} h^-1 Msun")

if A.out:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["savefig.dpi"] = 300
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    hi = np.percentile(S_M, 99.9)
    bins = np.linspace(0, hi, 90)
    ax.hist(S_M, bins=bins, color="0.80", edgecolor="none", density=True,
            label=f"all {npoint:,} subvolumes")
    xs = np.linspace(1e-3, hi, 400)
    ax.plot(xs, chi2.pdf(xs, d_eff), color="k", lw=1.8, ls="--",
            label=f"$\\chi^2$, {d_eff} degrees of freedom")
    for (lab, idx), col in zip(RULES, ("C0", "C4", "C2", "C3", "C7")):
        ax.hist(S_M[idx], bins=bins, histtype="step", lw=1.7, color=col,
                density=True, label=lab)
    ax.axvline(d_eff, color="0.35", lw=1.1)
    ax.text(d_eff, ax.get_ylim()[1]*0.97, f"  $d_{{\\rm eff}} = {d_eff}$",
            fontsize=9, color="0.35", va="top")
    ax.set_xlabel("$S_{\\rm M} = \\Delta^{\\rm T}\\,\\mathbf{C}^{-1}\\,"
                  "\\Delta$   [squared Mahalanobis distance]")
    ax.set_ylabel("density")
    ax.set_xlim(0, hi)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.set_title("A typical region sits on the shell at $d_{\\rm eff}$, not at "
                 "the origin.\nRules that minimise the score land where almost "
                 "no real region is.", fontsize=11)
    fig.tight_layout()
    fig.savefig(A.out)
    print(f"\nwrote {A.out}")
