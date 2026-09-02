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
import glob, os, re, subprocess, time
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
_ap.add_argument("--ngrid", type=int, default=64, help="cells per side")
_ap.add_argument("--species", nargs="+", default=["matter", "cdm", "baryon"],
                 choices=["matter", "cdm", "baryon"],
                 help="which species to measure; one species costs a third of the FFTs")
_ap.add_argument("--npencils", type=int, default=0,
                 help="measure a random subset of this many pencils per realization "
                      "(0 = all 192). Pencils within one realization are correlated, so "
                      "more realizations buy more than more pencils per realization")
_ap.add_argument("--environment", action="store_true",
                 help="also record the tidal shear and mean overdensity of each pencil")
_ap.add_argument("--smooth", type=float, nargs="+", default=[20.0, 40.0],
                 help="Gaussian smoothing radii in Mpc/h for the tidal shear. Keep these "
                      "above the cell size and below the pencil width")
_ap.add_argument("--nthreads", type=int, default=None,
                 help="OpenMP threads for monofonIC. Deliberately has no default: the "
                      "right value depends on how many sweeps you intend to run at once")
_ap.add_argument("--compact", action="store_true",
                 help="write one file for the whole chunk instead of a directory per "
                      "seed. Required at large N_seed: per-seed directories put hundreds "
                      "of thousands of small files on the filesystem")
_ap.add_argument("--keep-fields", action="store_true",
                 help="retain each seed's delta(q) HDF5 (6 MB per seed)")
ARGS = _ap.parse_args()

if ARGS.nthreads is None:
    import platform
    ncpu = os.cpu_count() or 1
    detail = ""
    if platform.system() == "Darwin":
        try:
            import subprocess as _sp
            perf = int(_sp.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))
            eff = int(_sp.check_output(["sysctl", "-n", "hw.perflevel1.physicalcpu"]))
            detail = f" ({perf} performance + {eff} efficiency)"
            ncpu = perf
        except Exception:
            pass
    raise SystemExit(
        f"--nthreads is required.\n\n"
        f"This machine reports {os.cpu_count()} cores{detail}.\n"
        f"Only the monofonIC step is threaded; the measurement runs single-threaded in\n"
        f"numpy, so a single sweep leaves most of the machine idle. Splitting the seed\n"
        f"range across several sweeps run at once is usually the bigger win.\n\n"
        f"  one sweep alone      : --nthreads {ncpu}\n"
        f"  k sweeps in parallel : --nthreads {max(1, ncpu//4)}  (for k = 4)\n")
NTHREADS = ARGS.nthreads
SEEDS = list(range(ARGS.seed0, ARGS.seed0 + ARGS.nseeds))
NGRID, LBOX, FRAC = ARGS.ngrid, 700.0, 8   # pencil = 1/FRAC of the box in two axes
SPECIES = ARGS.species                          # dataset / theory column pairs below
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
    # CLASS occasionally faults inside its spline interpolation on startup (SIGBUS
    # in array_interpolate_spline). It is intermittent and a rerun clears it, so a
    # long sweep should not die on one bad draw.
    for attempt in range(3):
        with open(f"{rundir}/run.log", "w") as log:
            r = subprocess.run([BIN, conf], cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode == 0 and os.path.exists(out):
            return out
        print(f"   seed {seed}: monofonIC exited {r.returncode}, retrying "
              f"({attempt+1}/3)", flush=True)
    raise RuntimeError(f"monofonIC failed three times for seed {seed}; see {rundir}/run.log")


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
K2 = kk**2; K2[0, 0, 0] = 1.0
KV = [KX, KY, KZ]
kbin = np.array([kk.ravel()[ib == i+1].mean() for i in range(nb)])
nmodes = np.array([(ib == i+1).sum() for i in range(nb)])
binned = lambda P: np.array([P.ravel()[ib == i+1].mean() for i in range(nb)])


def pencil_slice(axis, i, j):
    """Index tuple selecting one pencil. Slicing beats building a mask array: at
    N = 128 the 192 masks would be 3.2 GB."""
    idx = [None, None, None]
    idx[axis] = slice(None)
    rest = [a for a in range(3) if a != axis]
    idx[rest[0]] = slice(i*npen, (i+1)*npen)
    idx[rest[1]] = slice(j*npen, (j+1)*npen)
    return tuple(idx)


def masked(d, p):
    """The field with everything outside the pencil set to zero."""
    out = np.zeros_like(d)
    sl = pencil_slice(*p)
    out[sl] = d[sl]
    return out


ALL_PENCILS = [(a, i, j) for a in range(3) for i in range(FRAC) for j in range(FRAC)]
if ARGS.npencils and ARGS.npencils < len(ALL_PENCILS):
    _rng = np.random.default_rng(12345)
    PENCILS = [ALL_PENCILS[t] for t in
               sorted(_rng.choice(len(ALL_PENCILS), ARGS.npencils, replace=False))]
else:
    PENCILS = ALL_PENCILS
W0 = np.zeros((NGRID,)*3); W0[pencil_slice(*PENCILS[0])] = 1.0
fvol = W0.mean()

# ---- theory and the window-convolved theory ---------------------------------
th = np.loadtxt(paths.REF_POWERSPEC)
kth = th[:, 0]
g = kk > 0
Wk2 = np.abs(np.fft.fftn(W0)/NGRID**3)**2
fWk2 = np.fft.fftn(Wk2)
P_theory, P_win = [], []
for sp in SPECIES:                                # (2 pi)^3: monofonIC's internal units
    Pth1 = th[:, THCOL[sp]]*(2*np.pi)**3
    Pth3 = np.zeros_like(kk)
    Pth3[g] = np.exp(np.interp(np.log(kk[g]), np.log(kth), np.log(Pth1)))
    P_theory.append(binned(Pth3))
    P_win.append(binned(np.real(np.fft.ifftn(np.fft.fftn(Pth3)*fWk2))/fvol))
P_theory, P_win = np.array(P_theory), np.array(P_win)

# Parallel array tasks share one --out, so only the first to arrive writes the
# shared theory file; the rest would collide on the HDF5 lock. Each chunk also
# carries its own copy, so a chunk is self-contained.
try:
    if not os.path.exists(f"{OUT}/theory.hdf5"):
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
except (OSError, BlockingIOError):
    pass   # another task got there first
print(f"pencil {npen}^2 x {NGRID} cells = {lperp:.1f}^2 x {LBOX:.0f} Mpc/h, f = {fvol:.6f}")
print(f"dk_perp = {dkperp:.4f} h/Mpc = {dkperp/kf:.0f} k_fund, k_Ny = {kny:.4f}, {nb} bins")
print(f"{len(PENCILS)} pencils per seed, {len(SEEDS)} seeds, DoFixing = {ARGS.dofixing}\n")

# ---- sweep ------------------------------------------------------------------
fit = (kbin > 2*dkperp) & (kbin <= 0.9*kny)      # band used for the metrics
rows = []
ACC = {k: [] for k in ("seed", "P_full", "P_pencil", "shear", "dbar", "lambda",
                       "webtype", "contrast", "bulk")}
for s in SEEDS:
    t0 = time.time()
    # Unique per chunk: parallel array tasks share one --out directory, so a
    # single _work would have them deleting each other's field mid-run.
    rundir = f"{OUT}/_work_{SEEDS[0]:07d}" if ARGS.compact else f"{OUT}/seed_{s:05d}"
    os.makedirs(rundir, exist_ok=True)
    if ARGS.compact:
        for stale in glob.glob(f"{rundir}/deltaq.hdf5"):
            os.remove(stale)
    fic = run_ic(s, rundir)
    with h5py.File(fic) as f:
        d = {sp: f[DSET[sp]][:].astype(float) for sp in SPECIES}
        zstart = float(f["Header"].attrs["zstart"])
        Dplus = float(f["Header"].attrs["Dplus"])

    P_full = np.empty((len(SPECIES), nb))
    P_pen = np.empty((len(SPECIES), len(PENCILS), nb))
    for m, sp in enumerate(SPECIES):
        P_full[m] = binned(V*np.abs(np.fft.fftn(d[sp])/NGRID**3)**2)
        for n, p in enumerate(PENCILS):
            P_pen[m, n] = binned(V*np.abs(np.fft.fftn(masked(d[sp], p))/NGRID**3)**2/fvol)

    # Physical state of each region: the tidal shear s_ij = (d_i d_j / laplacian
    # - delta_ij/3) delta, and the mean overdensity, both on the smoothed field.
    # For a Gaussian field <s^2> = (2/3) <delta^2>, which the run prints as a check.
    if ARGS.environment:
        nR = len(ARGS.smooth)
        shear = np.empty((nR, len(PENCILS)))
        dbar = np.empty((nR, len(PENCILS)))
        lam = np.empty((nR, len(PENCILS), 3))     # mean sorted eigenvalues per pencil
        web = np.empty((nR, len(PENCILS), 4))     # knot, filament, sheet, void fractions
        contrast = np.empty((nR, len(PENCILS)))   # region minus its surrounding tiles
        bulk = np.empty((len(PENCILS), 3))        # mean Zel'dovich displacement, Mpc/h
        dk0 = np.fft.fftn(d["matter"])
        # Bulk flow: the region's mean displacement, Psi(k) = i k delta / k^2. It is
        # weighted to large scales by the 1/k, which is where the criterion acts.
        psi = [np.real(np.fft.ifftn(1j*KV[a]/K2*dk0)) for a in range(3)]
        for n, p in enumerate(PENCILS):
            sl = pencil_slice(*p)
            bulk[n] = [psi[a][sl].mean() for a in range(3)]
        del psi
        for r, R in enumerate(ARGS.smooth):
            dks = dk0*np.exp(-0.5*K2*R**2)
            dsm = np.real(np.fft.ifftn(dks))
            # The full tidal tensor T_ij = d_i d_j Phi, with laplacian Phi = delta.
            # Its eigenvalues sum to delta and decide the collapse geometry; the
            # traceless part of it is the shear.
            T = np.empty(dsm.shape + (3, 3))
            for a in range(3):
                for b in range(a, 3):
                    T[..., a, b] = T[..., b, a] = np.real(np.fft.ifftn(KV[a]*KV[b]/K2*dks))
            # sum_ij (T_ij - delta_ij delta/3)^2 = sum_ij T_ij^2 - delta^2/3,
            # which avoids building the subtracted tensor.
            s2 = (T*T).sum((-1, -2)) - dsm*dsm/3.0
            if s == SEEDS[0] and r == 0:
                print(f"   check <s^2>/<delta^2> = {s2.mean()/dsm.var():.4f} (2/3 expected)")
            # eigvalsh returns ascending; reverse so lam[...,0] is the largest, the
            # axis along which a region collapses first (Zel'dovich).
            ev = np.linalg.eigvalsh(T)[..., ::-1]
            npos = (ev > 0.0).sum(-1)             # T-web class: 3 knot ... 0 void
            for n, p in enumerate(PENCILS):
                sl = pencil_slice(*p)
                shear[r, n] = np.sqrt(s2[sl].mean())
                dbar[r, n] = dsm[sl].mean()
                lam[r, n] = ev[sl].mean((0, 1, 2))
                cnt = np.bincount(npos[sl].ravel(), minlength=4)
                web[r, n] = cnt[::-1]/cnt.sum()      # knot, filament, sheet, void
                # Environment contrast: the region against the eight tiles around
                # it, which is what "sits in an overdense environment" means for a
                # pencil. The box mean is the wrong reference for that question.
                ax, ti, tj = p
                ring = [((ti+di) % FRAC, (tj+dj) % FRAC)
                        for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
                contrast[r, n] = dbar[r, n] - np.mean(
                    [dsm[pencil_slice(ax, a_, b_)].mean() for a_, b_ in ring])

    if ARGS.compact:
        ACC["seed"].append(s)
        ACC["P_full"].append(P_full)
        ACC["P_pencil"].append(P_pen)
        if ARGS.environment:
            ACC["shear"].append(shear); ACC["dbar"].append(dbar)
            ACC["lambda"].append(lam); ACC["webtype"].append(web)
            ACC["contrast"].append(contrast); ACC["bulk"].append(bulk)
    with h5py.File(f"{rundir}/pk.hdf5", "w") as f:
        f["k"] = kbin
        f["P_full"] = P_full
        f["P_pencil"] = P_pen
        if ARGS.environment:
            f["shear"] = shear
            f["dbar"] = dbar
            f["lambda"] = lam
            f["webtype"] = web          # fractions, order: knot, filament, sheet, void
            f["contrast"] = contrast
            f["bulk"] = bulk
            f["smooth_R"] = np.array(ARGS.smooth)
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
    if not ARGS.keep_fields and os.path.exists(fic):
        os.remove(fic)
    print(f"seed {s}: full-box P/theory median {np.median(P_full[0][fit]/P_theory[0][fit]):.4f}, "
          f"mean pencil/P_win {np.median(P_pen[0][:, fit].mean(0)/P_win[0][fit]):.4f}, "
          f"{time.time()-t0:.1f} s")

if ARGS.compact:
    tag = f"{SEEDS[0]:06d}_{len(SEEDS):05d}"
    with h5py.File(f"{OUT}/chunk_{tag}.hdf5", "w") as f:
        f["k"] = kbin
        for key, v in ACC.items():
            if v:
                f[key] = np.array(v)
        f["pencil_axis"] = np.array([p[0] for p in PENCILS])
        f["pencil_i"] = np.array([p[1] for p in PENCILS])
        f["pencil_j"] = np.array([p[2] for p in PENCILS])
        f["P_theory"] = P_theory      # self-contained: no shared file needed
        f["P_win"] = P_win
        if ARGS.environment:
            f["smooth_R"] = np.array(ARGS.smooth)
        f.attrs["species"] = np.array(SPECIES, dtype=h5py.string_dtype())
        f.attrs.update(dict(N=NGRID, L=LBOX, frac=FRAC, fvol=fvol, kny=kny,
                            dkperp=dkperp, dofixing=ARGS.dofixing,
                            nseeds=len(SEEDS), seed0=SEEDS[0]))
    import shutil
    shutil.rmtree(f"{OUT}/_work_{SEEDS[0]:07d}", ignore_errors=True)
    print(f"wrote {OUT}/chunk_{tag}.hdf5")

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
