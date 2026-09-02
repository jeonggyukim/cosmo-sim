"""Summary figure for a pencil seed sweep.

Top row: the full periodic box. Bottom row: the pencil subregions.
Left column: P(k). Right column: ratio to the unconvolved theory.
Individual realisations are thin grey lines; the mean is the thick red line.
"""
import argparse, glob, numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
import os, paths
ap.add_argument("--data", default=os.path.join(paths.DATA, "pencil_sweep_n64_L700"))
ap.add_argument("--png", default=None)
ap.add_argument("--nshow", type=int, default=250, help="individual pencil curves drawn")
ap.add_argument("--species", default="matter", choices=["matter", "cdm", "baryon"])
ap.add_argument("--nbest", type=int, default=10,
                help="highlight this many pencils that best match the RAW theory")
A = ap.parse_args()
SPLAB = {"matter": r"$\delta_{\rm m}$ (total matter)", "cdm": r"$\delta_{\rm c}$ (CDM)",
         "baryon": r"$\delta_{\rm b}$ (baryons)"}
PNG = A.png or os.path.join(
    paths.FIGS, f"sweep_summary_{os.path.basename(A.data.rstrip(os.sep))}_{A.species}.png")

with h5py.File(f"{A.data}/theory.hdf5") as f:
    sp_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = sp_names.index(A.species)
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    kny, dkperp, L, N = (f.attrs[x] for x in ("kny", "dkperp", "L", "N"))

P_full, P_pen, seeds, tags = [], [], [], []
for fn in sorted(glob.glob(f"{A.data}/seed_*/pk.hdf5")):
    with h5py.File(fn) as f:
        P_full.append(f["P_full"][SP]); P_pen.append(f["P_pencil"][SP])
        seeds.append(int(f.attrs["seed"]))
        tags.append(np.stack([f["pencil_axis"][:], f["pencil_i"][:], f["pencil_j"][:]], 1))
        zstart, Dplus = float(f.attrs["zstart"]), float(f.attrs["Dplus"])
        dofix = f.attrs.get("dofixing", "yes")
        dofix = dofix.decode() if isinstance(dofix, bytes) else str(dofix)
tags = np.concatenate(tags)
P_full = np.array(P_full)
P_pen = np.concatenate(P_pen)
nseed = len(seeds)

mf, mp = P_full.mean(0), P_pen.mean(0)
npen = len(P_pen)//len(seeds)
print(f"{nseed} seeds x {npen} pencils = {len(P_pen)} pencils")
print(f"full box : across-seed spread {np.abs(P_full/mf - 1).max():.2e}  (DoFixing = {dofix})")
print(f"pencils  : mean/P_win median {np.median(mp/P_win):.4f}; "
      f"per-pencil scatter of P/P_theory at k={k[0]:.3f}: {np.std(P_pen[:,0]/P_th[0]):.3f}")

# The nbest pencils that come closest to the RAW theory over the full band.
# These are the ones a seed search would select; each is an upper-tail draw whose
# realisation scatter happens to cancel the deterministic window deficit.
band = (k > 0) & (k <= 0.9*kny)
D_th = np.sqrt((np.log(P_pen[:, band]/P_th[band])**2).mean(1))
best = np.argsort(D_th)[:A.nbest]
seed_of = np.repeat(seeds, npen)
print(f"\nbest {A.nbest} pencils vs RAW theory over k = {k[band][0]:.4f}..{k[band][-1]:.4f}:")
for r, i in enumerate(best):
    print(f"  {r+1:2d}. seed {seed_of[i]}  axis {tags[i,0]} (i,j)=({tags[i,1]},{tags[i,2]})  "
          f"D = {D_th[i]:.4f} ({100*D_th[i]:.1f}%)   P/P_th at k_f = {P_pen[i,0]/P_th[0]:.2f}")
print(f"  median D over all {len(P_pen)} pencils: {np.median(D_th):.4f}\n")

rng = np.random.default_rng(0)
show = rng.choice(len(P_pen), size=min(A.nshow, len(P_pen)), replace=False)
GREY = dict(color="0.35", lw=0.5, alpha=0.3, zorder=1)
RED = dict(color="C3", lw=2.4, zorder=4)

fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
fig.subplots_adjust(hspace=0.10, wspace=0.24)

a = ax[0, 0]
for P in P_full:
    a.loglog(k, P, **GREY)
a.loglog(k, mf, **RED, label=f"mean of {nseed} seeds")
a.loglog(k, P_th, color="0.1", lw=1.2, ls="--", zorder=5, label="theory (CLASS, back-scaled)")
a.plot([], [], **GREY, label=f"individual seeds ({nseed})")
a.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
a.set_title(r"(a)  full periodic box", fontsize=10)
a.legend(frameon=False, fontsize=8, loc="lower left")

a = ax[0, 1]
for P in P_full:
    a.semilogx(k, P/P_th, **GREY)
a.semilogx(k, mf/P_th, **RED)
a.axhline(1.0, color="0.1", ls="--", lw=1.2)
a.set_ylim(0.9, 1.1)
a.set_ylabel(r"$P/P_{\rm theory}$")
a.set_title(rf"(b)  full box, seed to seed  ($\tt DoFixing={dofix}$)", fontsize=10)
a.set_ylim(0.0, 2.2) if dofix == "no" else a.set_ylim(0.9, 1.1)
a.text(0.03, 0.10, "$\\tt DoFixing=yes$ pins $|\\delta_k|$ to $\\sqrt{P(k)}$, so the seed\n"
                   f"sets phases only: across-seed spread is {np.abs(P_full/mf - 1).max():.0e}",
       transform=a.transAxes, fontsize=7.5, color="0.3")

a = ax[1, 0]
for i in show:
    a.loglog(k, P_pen[i], **GREY)
for i in best:
    a.loglog(k, P_pen[i], color="C0", lw=1.0, alpha=0.9, zorder=3)
a.plot([], [], color="C0", lw=1.0, label=f"{A.nbest} best matches to raw theory")
a.loglog(k, mp, **RED, label=f"mean of {len(P_pen)} pencils")
a.loglog(k, P_win, color="C1", lw=1.5, ls=(0, (6, 3)), zorder=6, label=r"theory $\ast$ pencil window")
a.loglog(k, P_th, "--", color="0.1", lw=1.2, zorder=5, label="theory")
a.plot([], [], **GREY, label=f"individual pencils ({len(show)} of {len(P_pen)} drawn)")
a.axvline(dkperp, ls="--", color="C1", lw=1)
a.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
a.set_title(r"(c)  pencil subregions, $\ell_\perp = L/8$", fontsize=10)
a.legend(frameon=False, fontsize=8, loc="lower left")

a = ax[1, 1]
for i in show:
    a.semilogx(k, P_pen[i]/P_th, **GREY)
for i in best:
    a.semilogx(k, P_pen[i]/P_th, color="C0", lw=1.0, alpha=0.9, zorder=3)
a.plot([], [], color="C0", lw=1.0, label=f"{A.nbest} best matches to raw theory")
a.semilogx(k, mp/P_th, **RED, label="mean of pencils")
a.semilogx(k, P_win/P_th, color="C1", lw=1.5, ls=(0, (6, 3)), zorder=6,
           label=r"$(\rm theory \ast window)/theory$: the expectation")
a.axhline(1.0, color="0.1", ls="--", lw=1.2)
a.axvline(dkperp, ls="--", color="C1", lw=1)
a.text(dkperp*1.06, 0.08, r"$\Delta k_\perp$", color="C1", fontsize=8)
a.set_ylim(0, 2.3)
a.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); a.set_ylabel(r"$P/P_{\rm theory}$")
a.set_title(r"(d)  pencils scatter about the window curve, not about 1", fontsize=10)
a.legend(frameon=False, fontsize=8, loc="upper right")

for a in ax.ravel():
    a.axvline(kny, ls=":", color="0.6", zorder=0)

fig.suptitle(rf"Seed sweep: {nseed} monofonIC realisations $\times$ {npen} disjoint pencils "
             rf"$= {len(P_pen)}$ pencil spectra"
             "\n"
             rf"$\delta_{{\rm m}}(q)$, $N={int(N)}^3$, $L={L:g}\,$Mpc$/h$, $z={zstart:g}$, "
             rf"$D_+={Dplus:.4g}$, $\tt DoFixing=yes$, 2LPT, CV_22; pencil $= {L/8:.0f}^2 \times {L:.0f}$ Mpc$/h$",
             fontsize=10.5, y=0.975)
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print("saved", PNG)
