"""Summary figure for a pencil seed sweep.

Row 1: the full periodic box. Row 2: the pencil subregions, in P(k). Row 3: the
same pencils in xi(r), when the sweep was run with --xi.
Left column: the measurement. Right column: its ratio to the unconvolved theory.
Individual realisations are thin grey lines; the mean is the thick red line.

The third row is the reason the first two matter. Masking multiplies in
configuration space, so the mask's own pair count divides out of xi exactly and
the pencils scatter about the theory itself. The same mask convolves P(k), which
no per-mode division undoes, so the pencils there scatter about the
window-convolved curve instead. Panels (d) and (f) are the same subvolumes
measured two ways, and only one of them needs a window drawn on it.
"""
import argparse, glob, os, numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
import paths
ap.add_argument("--data", default=os.path.join(paths.DATA, "pencil_sweep_n64_L700"))
ap.add_argument("--png", default=None)
ap.add_argument("--nshow", type=int, default=250, help="individual pencil curves drawn")
ap.add_argument("--species", default="matter", choices=["matter", "cdm", "baryon"])
ap.add_argument("--nbest", type=int, default=10,
                help="highlight this many pencils that best match the RAW theory")
ap.add_argument("--powerspec", default=None,
                help="monofonIC *_input_powerspec.txt for the xi theory curve. The "
                     "binned P(k) the sweep stores smooths the acoustic feature, which "
                     "biases xi_theory at small separations; the raw table does not")
A = ap.parse_args()
SPLAB = {"matter": r"$\delta_{\rm m}$ (total matter)", "cdm": r"$\delta_{\rm c}$ (CDM)",
         "baryon": r"$\delta_{\rm b}$ (baryons)"}
PNG = A.png or os.path.join(
    paths.FIGS, f"sweep_summary_{os.path.basename(A.data.rstrip(os.sep))}_{A.species}.png")

# Each chunk carries its own copy of the theory, so a directory of chunks needs
# no shared file; theory.hdf5 is only written by runs that keep one.
_ref = f"{A.data}/theory.hdf5"
if not os.path.exists(_ref):
    _hits = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
    if not _hits:
        raise SystemExit(f"no theory.hdf5 and no chunk_*.hdf5 under {A.data}")
    _ref = _hits[0]
with h5py.File(_ref) as f:
    sp_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = sp_names.index(A.species)
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    kny, dkperp, L, N = (f.attrs[x] for x in ("kny", "dkperp", "L", "N"))

# Two layouts: one directory per seed for small local runs, one chunk file per
# batch for cluster arrays. Both carry the same arrays.
P_full, P_pen, X_full, X_pen, seeds, tags = [], [], [], [], [], []
zstart = Dplus = None
chunks = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
per_seed = sorted(glob.glob(f"{A.data}/seed_*/pk.hdf5"))
rbin = None
for fn in (chunks or per_seed):
    with h5py.File(fn) as f:
        tag = np.stack([f["pencil_axis"][:], f["pencil_i"][:], f["pencil_j"][:]], 1)
        if chunks:
            P_full.append(f["P_full"][:, SP])
            pp = f["P_pencil"][:, SP]                 # (nseed, npencil, nk)
            P_pen.append(pp.reshape(-1, pp.shape[-1]))
            sd = f["seed"][:] if "seed" in f else np.arange(pp.shape[0])
            seeds.extend(int(x) for x in sd)
            tags.append(np.tile(tag, (pp.shape[0], 1)))
            if "xi_full" in f:
                rbin = f["r"][:]
                R_EDGES = (f["r_edges"][:] if "r_edges" in f
                           else np.linspace(0.0, 250.0, len(rbin) + 1))
                X_full.append(f["xi_full"][:, SP])
                xp = f["xi_pencil"][:, SP]
                X_pen.append(xp.reshape(-1, xp.shape[-1]))
        else:
            P_full.append(f["P_full"][SP]); P_pen.append(f["P_pencil"][SP])
            seeds.append(int(f.attrs["seed"]))
            tags.append(tag)
            if "xi_full" in f:
                rbin = f["r"][:]
                R_EDGES = (f["r_edges"][:] if "r_edges" in f
                           else np.linspace(0.0, 250.0, len(rbin) + 1))
                X_full.append(f["xi_full"][SP][None])
                X_pen.append(f["xi_pencil"][SP])
        if zstart is None:
            zstart = float(f.attrs.get("zstart", 200.0))
            Dplus = float(f.attrs.get("Dplus", np.nan))
        dofix = f.attrs.get("dofixing", "yes")
        dofix = dofix.decode() if isinstance(dofix, bytes) else str(dofix)
tags = np.concatenate(tags)
P_full = np.concatenate(P_full) if chunks else np.array(P_full)
P_pen = np.concatenate(P_pen)
HAVE_XI = bool(X_pen) and rbin is not None
if HAVE_XI:
    X_full = np.concatenate(X_full)
    X_pen = np.concatenate(X_pen)
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

# xi(r) of the linear theory, on the measurement's own separation bins. With
# P(k) = V <|delta_k|^2> and delta_k the transform divided by N^3,
# xi(r) = (1/V) sum_k P(k) exp(i k.r) = (N^3/V) ifftn(P)(r), which at r = 0 is
# the variance, as the measured curve is. Built from the finely sampled table
# when one is given: interpolating the sweep's own binned P(k) smooths the
# acoustic feature and leaves xi_theory high by a per cent at small separations.
def xi_theory_curve():
    Ni, Li = int(N), float(L)
    kf = 2*np.pi/Li
    ka = np.fft.fftfreq(Ni, d=1.0/Ni)*kf
    k3 = np.sqrt(sum(g**2 for g in np.meshgrid(ka, ka, ka, indexing="ij")))
    pos = k3 > 0
    if A.powerspec:
        th = np.loadtxt(A.powerspec)
        kk, PP = th[:, 0], th[:, {"matter": 1, "cdm": 2, "baryon": 3}[A.species]]*(2*np.pi)**3
    else:
        kk, PP = k, P_th
    P3 = np.zeros_like(k3)
    P3[pos] = np.exp(np.interp(np.log(k3[pos]), np.log(kk), np.log(PP)))
    x3 = np.real(np.fft.ifftn(P3))*Ni**3/Li**3
    ra = np.minimum(np.arange(Ni), Ni - np.arange(Ni))*(Li/Ni)
    r3 = np.sqrt(sum(g**2 for g in np.meshgrid(ra, ra, ra, indexing="ij")))
    edge = R_EDGES
    idx = np.digitize(r3.ravel(), edge) - 1
    inb = (idx >= 0) & (idx < len(rbin))
    cnt = np.bincount(idx[inb], minlength=len(rbin))[:len(rbin)]
    return (np.bincount(idx[inb], weights=x3.ravel()[inb], minlength=len(rbin))[:len(rbin)]
            / np.maximum(cnt, 1))


NROW = 3 if HAVE_XI else 2
fig, ax = plt.subplots(NROW, 2, figsize=(12.0, 4.0*NROW))
fig.subplots_adjust(hspace=0.28, wspace=0.24)

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

# Without sharex each panel draws its own minor tick labels, and on a log axis
# spanning less than two decades they collide into unreadable overlap.
from matplotlib.ticker import NullFormatter
for a in ax.ravel():
    a.xaxis.set_minor_formatter(NullFormatter())
for a in ax[:2].ravel():
    a.axvline(kny, ls=":", color="0.6", zorder=0)
    a.set_xlim(k.min(), k.max())

if HAVE_XI:
    xi_th = xi_theory_curve()
    lperp = L/8.0
    # Only where the theory is safely away from its zero crossing near 80 Mpc/h:
    # a ratio to a vanishing denominator measures the crossing, not the estimator.
    gx = (rbin > 0) & (rbin <= lperp) & (xi_th > 0)
    mx = X_pen.mean(0)
    print(f"\npencils : xi mean/theory over r < {lperp:.0f} Mpc/h "
          f"{np.average((mx/xi_th)[gx], weights=np.abs(xi_th[gx])):.4f}; "
          f"box {np.average((X_full.mean(0)/xi_th)[gx], weights=np.abs(xi_th[gx])):.4f}")

    a = ax[2, 0]
    for i in show:
        pos_i = gx & (X_pen[i] > 0)
        a.loglog(rbin[pos_i], X_pen[i][pos_i], **GREY)
    for i in best:
        pos_i = gx & (X_pen[i] > 0)
        a.loglog(rbin[pos_i], X_pen[i][pos_i], color="C0", lw=1.0, alpha=0.9, zorder=3)
    a.plot([], [], color="C0", lw=1.0, label=f"{A.nbest} best matches to raw theory")
    a.loglog(rbin[gx], mx[gx], **RED, label=f"mean of {len(X_pen)} pencils")
    a.loglog(rbin[gx], xi_th[gx], "--", color="0.1", lw=1.2, zorder=5,
             label="theory (no window: none applies)")
    a.plot([], [], **GREY, label=f"individual pencils ({len(show)} drawn)")
    a.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); a.set_ylabel(r"$\xi(r)$")
    a.set_title(r"(e)  the same pencils, in configuration space", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower left")

    a = ax[2, 1]
    for i in show:
        a.semilogx(rbin[gx], (X_pen[i]/xi_th)[gx], **GREY)
    for i in best:
        a.semilogx(rbin[gx], (X_pen[i]/xi_th)[gx], color="C0", lw=1.0, alpha=0.9, zorder=3)
    a.semilogx(rbin[gx], (mx/xi_th)[gx], **RED, label="mean of pencils")
    a.axhline(1.0, color="0.1", ls="--", lw=1.2, label="theory: the expectation, unconvolved")
    a.set_ylim(0, 2.3)
    a.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); a.set_ylabel(r"$\xi/\xi_{\rm theory}$")
    a.set_title(r"(f)  pencils scatter about 1: there is no window to subtract",
                fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    for a in ax[2]:
        a.axvline(lperp, ls="--", color="C1", lw=1)
        a.set_xlim(rbin[gx].min(), lperp)

fig.suptitle(rf"Seed sweep: {nseed} monofonIC realisations $\times$ {npen} disjoint pencils "
             rf"$= {len(P_pen)}$ pencil spectra"
             "\n"
             + SPLAB[A.species]
             + rf", $N={int(N)}^3$, $L={L:g}\,$Mpc$/h$, $z={zstart:g}$, "
             + (rf"$D_+={Dplus:.4g}$, " if np.isfinite(Dplus) else "")
             + rf"$\tt DoFixing={dofix}$, 2LPT, CV_22; "
             + rf"pencil $= {L/8:.0f}^2 \times {L:.0f}$ Mpc$/h$",
             fontsize=10.5, y=0.975)
fig.savefig(PNG, dpi=300, bbox_inches="tight")
print("saved", PNG)
