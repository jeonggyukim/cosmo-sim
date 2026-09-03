"""Distribution of how far individual pencils sit from the two reference curves.

For every pencil, the deviation over a band of k is summarised as

    D = rms over the band of  ln( P_pencil(k) / P_ref(k) )

with P_ref either the raw back-scaled CLASS spectrum, or that spectrum convolved
with the pencil window, which is the expectation of the pencil estimator. The
gap between the two histograms is the part of the mismatch that no choice of
seed can remove, because it is a property of the geometry rather than of the
realisation.
"""
import argparse, glob, numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
import os, paths
ap.add_argument("--data", default=os.path.join(paths.DATA, "pencil_sweep_n64_L700_x30"))
ap.add_argument("--png", default=None)
ap.add_argument("--species", default="matter", choices=["matter", "cdm", "baryon"])
A = ap.parse_args()
PNG = A.png or os.path.join(
    paths.FIGS, f"deviation_stats_{os.path.basename(A.data.rstrip(os.sep))}_{A.species}.png")

# Two layouts: one directory per seed for small local runs, one chunk file per
# batch for cluster arrays. Each chunk carries its own copy of the theory, so a
# directory of chunks needs no shared theory.hdf5.
chunks = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
per_seed = sorted(glob.glob(f"{A.data}/seed_*/pk.hdf5"))
if not chunks and not per_seed:
    raise SystemExit(f"no chunk_*.hdf5 or seed_*/pk.hdf5 under {A.data}")

ref = chunks[0] if chunks else f"{A.data}/theory.hdf5"
with h5py.File(ref) as f:
    sp_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = sp_names.index(A.species)
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    kny, dkperp, L, N = (f.attrs[x] for x in ("kny", "dkperp", "L", "N"))

P_pen, seeds = [], []
zstart, Dplus = 200.0, float("nan")
for fn in (chunks or per_seed):
    with h5py.File(fn) as f:
        if chunks:
            pp = f["P_pencil"][:, SP]              # (nseed, npencil, nk)
            P_pen.append(pp.reshape(-1, pp.shape[-1]))
            seeds.extend(int(x) for x in f["seed"][:])
        else:
            P_pen.append(f["P_pencil"][SP])
            seeds.append(int(f.attrs["seed"]))
            zstart = float(f.attrs["zstart"]); Dplus = float(f.attrs["Dplus"])
        dofix = f.attrs.get("dofixing", "yes")
        dofix = dofix.decode() if isinstance(dofix, bytes) else str(dofix)
P_pen = np.concatenate(P_pen)
nseed, npen = len(seeds), len(P_pen)//len(seeds)
print(f"{nseed} seeds x {npen} pencils = {len(P_pen)} pencil spectra")

BANDS = [("all $k$", (k > 0) & (k <= 0.9*kny)),
         (r"$k \leq 2\Delta k_\perp$", k <= 2*dkperp),
         (r"$k > 2\Delta k_\perp$", (k > 2*dkperp) & (k <= 0.9*kny))]

fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
fig.subplots_adjust(hspace=0.30, wspace=0.24)
axes = [ax[0, 0], ax[0, 1], ax[1, 0]]
CT, CW = "0.35", "C1"
cum = {}

for (name, sel), a in zip(BANDS, axes):
    dth = np.sqrt((np.log(P_pen[:, sel]/P_th[sel])**2).mean(1))
    dwin = np.sqrt((np.log(P_pen[:, sel]/P_win[sel])**2).mean(1))
    cum[name] = (dth, dwin)
    wdef = P_win[sel]/P_th[sel]
    bins = np.linspace(0, max(np.percentile(dth, 99.5), np.percentile(dwin, 99.5)), 60)
    a.hist(dth, bins=bins, color=CT, alpha=0.55, label="vs theory")
    a.hist(dwin, bins=bins, histtype="step", color=CW, lw=2.0, label=r"vs theory $\ast$ window")
    for v, c in ((np.median(dth), CT), (np.median(dwin), CW)):
        a.axvline(v, color=c, ls="--", lw=1.2)
    a.set_xlabel(r"$D = \mathrm{rms}\ \ln(P_{\rm pencil}/P_{\rm ref})$ over the band")
    a.set_ylabel("pencils")
    a.set_title(f"{name}: {sel.sum()} bins, "
                rf"window deficit ${wdef.min():.2f}$–${wdef.max():.2f}$", fontsize=9.5)
    a.legend(frameon=False, fontsize=8)
    a.text(0.97, 0.55, f"median vs theory  {np.median(dth):.3f}\n"
                       f"median vs window  {np.median(dwin):.3f}\n"
                       f"best vs theory    {dth.min():.3f}",
           transform=a.transAxes, ha="right", va="top", fontsize=7.5,
           family="monospace", color="0.25")

a = ax[1, 1]
thr = np.logspace(np.log10(0.005), np.log10(1.0), 200)
for (name, _), ls in zip(BANDS, ["-", "--", ":"]):
    dth, dwin = cum[name]
    a.loglog(thr, [(dth < t).mean() for t in thr], ls, color=CT, lw=1.8, label=f"vs theory, {name}")
    a.loglog(thr, [(dwin < t).mean() for t in thr], ls, color=CW, lw=1.8, label=rf"vs window, {name}")
a.axhline(1.0/len(P_pen), color="0.6", lw=1, ls="-")
a.text(0.006, 1.3/len(P_pen), f"1 pencil in {len(P_pen)}", fontsize=7.5, color="0.4")
a.set_xlabel(r"threshold on $D$")
a.set_ylabel("fraction of pencils below threshold")
a.set_title("(d)  hit rate: fraction of pencils matching to better than $D$", fontsize=9.5)
# One column, in the order the curves are drawn: each band's pair together,
# theory then window. Two columns filled the second before the first was read,
# which broke that pairing.
a.legend(fontsize=7, ncol=1, loc="upper left", framealpha=0.95)
a.set_ylim(0.5/len(P_pen), 1.5)

for lab, a in zip("abc", axes):
    a.set_title(f"({lab})  " + a.get_title(), fontsize=9.5)

fig.suptitle(rf"Deviation of individual pencils from theory and from theory $\ast$ window: "
             rf"{nseed} seeds $\times$ {npen} pencils $= {len(P_pen)}$ spectra"
             "\n"
             rf"$\delta_{{\rm m}}(q)$, $N={int(N)}^3$, $L={L:g}\,$Mpc$/h$, $z={zstart:g}$, "
             rf"$\tt DoFixing={dofix}$, 2LPT, {A.species}; pencil $={L/8:.0f}^2\times{L:.0f}$ Mpc$/h$, "
             rf"$\Delta k_\perp={dkperp:.3f}\,h/$Mpc",
             fontsize=10.5, y=0.97)
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print("saved", PNG)

for name, (dth, dwin) in cum.items():
    print(f"\n{name}: median D vs theory {np.median(dth):.4f}, vs window {np.median(dwin):.4f}")
    for t in (0.02, 0.05, 0.10):
        n1, n2 = (dth < t).sum(), (dwin < t).sum()
        print(f"   D < {t:.2f}: vs theory {n1:6d}/{len(dth)} ({100*n1/len(dth):6.3f}%)"
              f"   vs window {n2:6d}/{len(dwin)} ({100*n2/len(dwin):6.3f}%)")
