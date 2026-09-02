"""Sweep monofonIC seeds; save full-box and per-pencil P(k) for later analysis.

For each seed: generate the Lagrangian density field delta(q) with monofonIC's
LagrangianDensityOnly path, then measure

  * P_full(k)   -- the whole periodic box, no window
  * P_pencil(k) -- 3 orientations x (FRAC^2) disjoint pencils, each 1/FRAC of the
                   box in two axes and the full box in the third, by masked FFT
                   with the P = V|F|^2/f normalisation (Park et al. 1994 eq. 13
                   with w_j = 1 and no shot-noise term, the field being a grid).

The theory file holds both comparison curves on the same k bins: the raw
back-scaled CLASS spectrum, and that spectrum convolved with the pencil window,
which is the expectation of the pencil estimator. Judging a pencil against the
raw theory measures its realisation scatter AGAINST a deterministic ~30% window
deficit at low k; judging it against the convolved curve measures scatter alone.

Outputs under OUT/:
  theory.hdf5              k, P_theory, P_win, |W_k|^2 diagnostics
  seed_%05d/deltaq.hdf5    the delta(q) field (kept: 6 MB each, allows re-analysis)
  seed_%05d/pk.hdf5        k, P_full[3, nbins], P_pencil[3, 192, nbins], pencil axis/i/j
                           (species axis: matter, cdm, baryon -- see the "species" attr)
  summary.hdf5             per-(seed, pencil) deviation metrics vs both curves
"""
import os, re, subprocess, time
import numpy as np, h5py
import paths

BIN, TPL, OUT = paths.BIN, paths.REF_CONF, None

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--seed0", type=int, default=1001)
_ap.add_argument("--nseeds", type=int, default=10)
_ap.add_argument("--out", default=os.path.join(paths.DATA, "pencil_sweep_n64_L700"))
_ap.add_argument("--dofixing", choices=["yes", "no"], default="no",
                 help="Angulo & Pontzen amplitude fixing. no (the default, and what is used "
                      "in practice) gives Rayleigh-distributed amplitudes; yes pins "
                      "|delta_k| to sqrt(P(k)), so the seed then sets phases only and every "
                      "seed has an identical full-box P(k)")
_ap.add_argument("--keep-fields", action="store_true",
                 help="retain each seed's delta(q) HDF5 (6 MB per seed)")
ARGS = _ap.parse_args()
SEEDS = list(range(ARGS.seed0, ARGS.seed0 + ARGS.nseeds))
NGRID, LBOX, FRAC = 64, 700.0, 8      # pencil = 1/FRAC of the box in two axes
NTHREADS = 18
SPECIES = ["matter", "cdm", "baryon"]          # dataset / theory column pairs below
DSET = {"matter": "delta_q", "cdm": "delta_q_cdm", "baryon": "delta_q_baryon"}
THCOL = {"matter": 1, "cdm": 2, "baryon": 3}   # columns of *_input_powerspec.txt

OUT = ARGS.out
paths.require(paths.REF_CONF, paths.REF_POWERSPEC, binary=True)
os.makedirs(OUT, exist_ok=True)
tpl = open(TPL).read()


def run_ic(seed, rundir):
    """Generate delta(q) for one seed. Returns the path to the field file."""
    out = f"{rundir}/deltaq.hdf5"
    if os.path.exists(out):
        return out
    conf = f"{rundir}/deltaq.conf"
    c = tpl
    c = re.sub(r"^GridRes.*$", f"GridRes         = {NGRID}", c, flags=re.M)
    c = re.sub(r"^BoxLength.*$", f"BoxLength       = {LBOX:g}", c, flags=re.M)
    c = re.sub(r"^seed.*$", f"seed            = {seed}", c, flags=re.M)
    c = re.sub(r"^NumThreads.*$", f"NumThreads      = {NTHREADS}", c, flags=re.M)
    c = re.sub(r"^DoFixing.*$", f"DoFixing        = {ARGS.dofixing}", c, flags=re.M)
    c = re.sub(r"^filename.*$", f"filename        = {out}", c, flags=re.M)
    open(conf, "w").write(c)
    with open(f"{rundir}/run.log", "w") as log:
        subprocess.run([BIN, conf], cwd=rundir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


# ---- k grid, bins, pencil masks (identical for every seed) -------------------
kf, kny = 2*np.pi/LBOX, np.pi*NGRID/LBOX
npen, dx = NGRID//FRAC, LBOX/NGRID
lperp, dkperp = LBOX/FRAC, 2*np.pi*FRAC/LBOX
V = LBOX**3

kax = np.fft.fftfreq(NGRID, 1.0/NGRID)*kf
KX, KY, KZ = np.meshgrid(kax, kax, kax, indexing="ij")
kk = np.sqrt(KX**2 + KY**2 + KZ**2)
edges = np.arange(0.5, NGRID/2+1.0)*kf
ib = np.digitize(kk.ravel(), edges)
nb = len(edges)-1
kbin = np.array([kk.ravel()[ib == i+1].mean() for i in range(nb)])
nmodes = np.array([(ib == i+1).sum() for i in range(nb)])
binned = lambda P: np.array([P.ravel()[ib == i+1].mean() for i in range(nb)])


def pencil_mask(axis, i, j):
    W = np.zeros((NGRID,)*3)
    idx = [None, None, None]
    idx[axis] = slice(None)
    rest = [a for a in range(3) if a != axis]
    idx[rest[0]] = slice(i*npen, (i+1)*npen)
    idx[rest[1]] = slice(j*npen, (j+1)*npen)
    W[tuple(idx)] = 1.0
    return W


PENCILS = [(a, i, j) for a in range(3) for i in range(FRAC) for j in range(FRAC)]
MASKS = [pencil_mask(*p) for p in PENCILS]
fvol = MASKS[0].mean()

# ---- theory and the window-convolved theory ---------------------------------
th = np.loadtxt(paths.REF_POWERSPEC)
kth = th[:, 0]
g = kk > 0
Wk2 = np.abs(np.fft.fftn(MASKS[0])/NGRID**3)**2
fWk2 = np.fft.fftn(Wk2)
P_theory, P_win = [], []
for sp in SPECIES:                                # (2 pi)^3: monofonIC's internal units
    Pth1 = th[:, THCOL[sp]]*(2*np.pi)**3
    Pth3 = np.zeros_like(kk)
    Pth3[g] = np.exp(np.interp(np.log(kk[g]), np.log(kth), np.log(Pth1)))
    P_theory.append(binned(Pth3))
    P_win.append(binned(np.real(np.fft.ifftn(np.fft.fftn(Pth3)*fWk2))/fvol))
P_theory, P_win = np.array(P_theory), np.array(P_win)

with h5py.File(f"{OUT}/theory.hdf5", "w") as f:
    f["k"] = kbin
    f["nmodes"] = nmodes
    f["P_theory"] = P_theory
    f["P_win"] = P_win
    f.attrs["species"] = np.array(SPECIES, dtype=h5py.string_dtype())
    for key, val in dict(N=NGRID, L=LBOX, frac=FRAC, npen=npen, lperp=lperp,
                         dkperp=dkperp, kny=kny, kf=kf, fvol=fvol,
                         npencils=len(PENCILS)).items():
        f.attrs[key] = val
print(f"pencil {npen}^2 x {NGRID} cells = {lperp:.1f}^2 x {LBOX:.0f} Mpc/h, f = {fvol:.6f}")
print(f"dk_perp = {dkperp:.4f} h/Mpc = {dkperp/kf:.0f} k_fund, k_Ny = {kny:.4f}, {nb} bins")
print(f"{len(PENCILS)} pencils per seed, {len(SEEDS)} seeds, DoFixing = {ARGS.dofixing}\n")

# ---- sweep ------------------------------------------------------------------
fit = (kbin > 2*dkperp) & (kbin <= 0.9*kny)      # band used for the metrics
rows = []
for s in SEEDS:
    t0 = time.time()
    rundir = f"{OUT}/seed_{s:05d}"
    os.makedirs(rundir, exist_ok=True)
    fic = run_ic(s, rundir)
    with h5py.File(fic) as f:
        d = {sp: f[DSET[sp]][:].astype(float) for sp in SPECIES}
        zstart = float(f["Header"].attrs["zstart"])
        Dplus = float(f["Header"].attrs["Dplus"])

    P_full = np.empty((len(SPECIES), nb))
    P_pen = np.empty((len(SPECIES), len(PENCILS), nb))
    for m, sp in enumerate(SPECIES):
        P_full[m] = binned(V*np.abs(np.fft.fftn(d[sp])/NGRID**3)**2)
        for n, W in enumerate(MASKS):
            P_pen[m, n] = binned(V*np.abs(np.fft.fftn(W*d[sp])/NGRID**3)**2/fvol)

    with h5py.File(f"{rundir}/pk.hdf5", "w") as f:
        f["k"] = kbin
        f["P_full"] = P_full
        f["P_pencil"] = P_pen
        f["pencil_axis"] = np.array([p[0] for p in PENCILS])
        f["pencil_i"] = np.array([p[1] for p in PENCILS])
        f["pencil_j"] = np.array([p[2] for p in PENCILS])
        f.attrs.update(dict(seed=s, N=NGRID, L=LBOX, zstart=zstart, Dplus=Dplus,
                            frac=FRAC, fvol=fvol, kny=kny, dkperp=dkperp,
                            dofixing=ARGS.dofixing))
        f.attrs["species"] = np.array(SPECIES, dtype=h5py.string_dtype())

    # deviation of each pencil from each reference curve, in the fit band
    for m, sp in enumerate(SPECIES):
        lr_th = np.log(P_pen[m][:, fit]/P_theory[m][fit])
        lr_win = np.log(P_pen[m][:, fit]/P_win[m][fit])
        for n, p in enumerate(PENCILS):
            rows.append((s, m, p[0], p[1], p[2],
                         np.sqrt((lr_th[n]**2).mean()), lr_th[n].mean(),
                         np.sqrt((lr_win[n]**2).mean()), lr_win[n].mean()))
    if not ARGS.keep_fields:
        os.remove(fic)
    print(f"seed {s}: full-box P/theory median {np.median(P_full[0][fit]/P_theory[0][fit]):.4f}, "
          f"mean pencil/P_win {np.median(P_pen[0][:, fit].mean(0)/P_win[0][fit]):.4f}, "
          f"{time.time()-t0:.1f} s")

rows = np.array(rows)
with h5py.File(f"{OUT}/summary.hdf5", "w") as f:
    for n, key in enumerate(["seed", "species", "axis", "i", "j",
                             "rms_logratio_theory", "mean_logratio_theory",
                             "rms_logratio_win", "mean_logratio_win"]):
        f[key] = rows[:, n]
    f.attrs["fit_kmin"] = kbin[fit][0]
    f.attrs["fit_kmax"] = kbin[fit][-1]
    f.attrs["species"] = np.array(SPECIES, dtype=h5py.string_dtype())
    f.attrs["note"] = ("rms_logratio_* is the rms of ln(P_pencil/P_ref) over the fit band; "
                       "ref = raw theory, or theory convolved with the pencil window; "
                       "the species column indexes the species attribute")
print(f"\nwrote {OUT}/{{theory,summary}}.hdf5 and {len(SEEDS)} seed directories")
