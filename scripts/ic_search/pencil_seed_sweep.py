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
import glob, os, random, re, subprocess, time
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
_ap.add_argument("--smooth-frac", type=float, nargs="+", default=None,
                 help="smoothing radii as fractions of the pencil's transverse width, "
                      "which is L/FRAC. Overrides --smooth. Given as a fraction the same "
                      "geometry holds at any box size, whereas a radius in Mpc/h that is "
                      "well inside the region for one box can exceed it for another")
_ap.add_argument("--xi", action="store_true",
                 help="also measure the two-point correlation function, for the whole "
                      "box and for each pencil. Masking multiplies in configuration "
                      "space, so the mask autocorrelation divides out exactly and a "
                      "subvolume xi is unbiased for the true xi, unlike its P(k)")
_ap.add_argument("--rmax", type=float, default=200.0,
                 help="largest separation for xi, in Mpc/h")
_ap.add_argument("--nrbin", type=int, default=40, help="number of separation bins")
_ap.add_argument("--interior-margin", type=float, default=0.0, metavar="M",
                 help="also measure every environment quantity on the pencil trimmed by "
                      "M smoothing radii from each long face. The smoothed field near a "
                      "face is built partly from material outside the pencil, so the "
                      "untrimmed average is not a property of the pencil alone. 0 "
                      "measures only the whole pencil")
_ap.add_argument("--powerspec", default=None, metavar="FILE",
                 help="monofonIC *_input_powerspec.txt to take the theory from. The "
                      "default reference table was written by a back-scaled run and is "
                      "the right theory only for runs that back-scale too, so a run "
                      "with --ztarget needs the table that run itself wrote")
_ap.add_argument("--ztarget", default=None, metavar="Z",
                 help="monofonIC [cosmology] ztarget: a redshift, or the word zstart "
                      "to take the transfer functions at the starting redshift and "
                      "apply no back-scaling. Back-scaling builds the baryon and cold "
                      "dark matter fields from one matter field scaled by one growth "
                      "factor, so the two species come out nearly identical and a "
                      "comparison between them measures nothing. Only a forward run "
                      "separates them. Default leaves the template's value alone")
_ap.add_argument("--kernel-weight", action="store_true",
                 help="also measure every environment quantity with each cell weighted "
                      "by the fraction of its smoothing kernel that fell inside the "
                      "pencil. A Gaussian has no compact support, so trimming a margin "
                      "of M radii still admits a one-sided tail of erfc(M/sqrt 2)/2 -- "
                      "16 percent at M=1, 2.3 percent at M=2 -- while this weighting is "
                      "continuous, keeps the whole region, and stays defined at radii "
                      "where a margin would leave nothing")
_ap.add_argument("--source-split", type=int, default=0, metavar="NPEN",
                 help="for this many pencils per realization, split the region's tidal "
                      "field into the part sourced by matter inside the region and the "
                      "part sourced by matter outside it. Poisson is linear, so "
                      "T[delta] = T[M delta] + T[(1-M) delta] exactly, and the two "
                      "pieces say how much of a region's measured shear is a property "
                      "of the region at all. Costs 8 extra FFTs per pencil per radius, "
                      "so a few pencils over a few hundred realizations is enough")
_ap.add_argument("--nthreads", type=int, default=None,
                 help="OpenMP threads for monofonIC. Deliberately has no default: the "
                      "right value depends on how many sweeps you intend to run at once")
_ap.add_argument("--seed-list", default=None, metavar="FILE",
                 help="run the seeds listed in FILE (one integer per line) instead of a "
                      "contiguous range. Used to rerun seeds an earlier sweep skipped; "
                      "--list-start and --nseeds select the slice this task takes")
_ap.add_argument("--list-start", type=int, default=0,
                 help="index into --seed-list at which this task starts")
_ap.add_argument("--batch-seeds", type=int, default=1, metavar="B",
                 help="generate B seeds per monofonIC call, using its SeedCount. "
                      "The transfer function does not depend on the seed, so a batch "
                      "costs one CLASS evaluation instead of B. Only consecutive seeds "
                      "can share a call, so a --seed-list is batched over its runs of "
                      "consecutive values. B fields exist at once, about 50 MB each at "
                      "N = 128 with three species")
_ap.add_argument("--nretry", type=int, default=6,
                 help="attempts per seed before it is skipped. CLASS segfaults on "
                      "some startup reads when many tasks contend for its data "
                      "files, and most such seeds succeed on a later try")
_ap.add_argument("--flush-every", type=int, default=0, metavar="N",
                 help="with --compact, write a finished chunk file every N seeds "
                      "instead of one file at the end. Each file is closed before "
                      "the next is started, so a running sweep can be analysed and "
                      "a killed task keeps what it already wrote. 0 writes once")
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
NRETRY = ARGS.nretry
if ARGS.seed_list:
    _all = [int(x) for x in open(ARGS.seed_list).read().split()]
    SEEDS = _all[ARGS.list_start:ARGS.list_start + ARGS.nseeds]
    if not SEEDS:
        raise SystemExit(f"--list-start {ARGS.list_start} is past the end of "
                         f"{ARGS.seed_list} ({len(_all)} seeds)")
else:
    SEEDS = list(range(ARGS.seed0, ARGS.seed0 + ARGS.nseeds))
NGRID, LBOX, FRAC = ARGS.ngrid, 700.0, 8   # pencil = 1/FRAC of the box in two axes
if ARGS.smooth_frac:
    ARGS.smooth = [f*LBOX/FRAC for f in ARGS.smooth_frac]
SPECIES = ARGS.species                          # dataset / theory column pairs below
DSET = {"matter": "delta_q", "cdm": "delta_q_cdm", "baryon": "delta_q_baryon"}
THCOL = {"matter": 1, "cdm": 2, "baryon": 3}   # columns of *_input_powerspec.txt

OUT = ARGS.out
paths.require(paths.REF_CONF, paths.REF_POWERSPEC, binary=True)
os.makedirs(OUT, exist_ok=True)
tpl = open(TPL).read()
_m = re.search(r"^zstart\s*=\s*([0-9.eE+-]+)", tpl, flags=re.M)
ZSTART_TPL = float(_m.group(1)) if _m else 200.0


def eigvals_sym3(T):
    """Eigenvalues of a field of symmetric 3x3 matrices, largest first.

    numpy's eigvalsh loops over the 2.1 million matrices of a 128^3 grid one at
    a time and costs 0.61 s per smoothing radius; the closed form for a
    symmetric 3x3 vectorises over the whole grid and costs 0.13 s, agreeing to
    1.4e-15. The construction is the standard one: shift by the mean eigenvalue
    q, scale by p so the shifted matrix has eigenvalues in [-2, 2], and read the
    three roots off the cosine of a third of the arccosine of half its
    determinant.
    """
    q = np.trace(T, axis1=-2, axis2=-1)/3.0
    p1 = T[..., 0, 1]**2 + T[..., 0, 2]**2 + T[..., 1, 2]**2
    p2 = ((T[..., 0, 0] - q)**2 + (T[..., 1, 1] - q)**2
          + (T[..., 2, 2] - q)**2) + 2*p1
    p = np.sqrt(p2/6.0) + 1e-300          # p = 0 only if T is already diagonal
    B = (T - q[..., None, None]*np.eye(3))/p[..., None, None]
    detB = (B[..., 0, 0]*(B[..., 1, 1]*B[..., 2, 2] - B[..., 1, 2]*B[..., 2, 1])
            - B[..., 0, 1]*(B[..., 1, 0]*B[..., 2, 2] - B[..., 1, 2]*B[..., 2, 0])
            + B[..., 0, 2]*(B[..., 1, 0]*B[..., 2, 1] - B[..., 1, 1]*B[..., 2, 0]))
    phi = np.arccos(np.clip(detB/2.0, -1.0, 1.0))/3.0
    e1 = q + 2*p*np.cos(phi)
    e3 = q + 2*p*np.cos(phi + 2*np.pi/3)
    return np.stack([e1, 3*q - e1 - e3, e3], -1)   # trace is exactly 3q


class ICFailure(RuntimeError):
    """monofonIC would not produce a field for this seed, after retries.

    A few seeds in a thousand fault deterministically inside CLASS. The sweep
    skips them rather than aborting: one lost realization is a far smaller loss
    than the whole chunk, and a seed is only a label, so dropping one does not
    favour any kind of realization. The skipped seeds are recorded with the
    output so the rate can be checked.
    """


def run_ic(seeds, rundir):
    """Generate delta(q) for consecutive seeds in one monofonIC call.

    Returns {seed: path}. monofonIC writes one file per seed and names it after
    the seed once more than one is asked for, and leaves the configured name
    alone for a single seed.
    """
    seeds = list(seeds)
    # Every seed gets its own file, whether it came from a batch or from a
    # single-seed retry. Sharing one name across retries would let the first
    # measurement delete the file the next seed still needs.
    if len(seeds) > 1:
        base = f"{rundir}/deltaq.hdf5"          # monofonIC appends _seed<N>
        outs = {s: f"{rundir}/deltaq_seed{s}.hdf5" for s in seeds}
    else:
        base = f"{rundir}/deltaq_seed{seeds[0]}.hdf5"
        outs = {seeds[0]: base}
    if all(os.path.exists(p) for p in outs.values()):
        return outs
    conf = f"{rundir}/deltaq.conf"
    c = tpl
    c = re.sub(r"^GridRes.*$", f"GridRes         = {NGRID}", c, flags=re.M)
    c = re.sub(r"^BoxLength.*$", f"BoxLength       = {LBOX:g}", c, flags=re.M)
    c = re.sub(r"^seed.*$", f"seed            = {seeds[0]}", c, flags=re.M)
    c = re.sub(r"^NumThreads.*$", f"NumThreads      = {NTHREADS}", c, flags=re.M)
    c = re.sub(r"^DoFixing.*$", f"DoFixing        = {ARGS.dofixing}", c, flags=re.M)
    c = re.sub(r"^filename.*$", f"filename        = {base}", c, flags=re.M)
    if ARGS.ztarget is not None:
        # Every occurrence, not the first: a template carrying a second ztarget
        # further down would otherwise override the one written here, which
        # monofonIC reports only as a "Redeclaration overwrites" warning.
        zt = ZSTART_TPL if ARGS.ztarget == "zstart" else float(ARGS.ztarget)
        c = re.sub(r"^ztarget.*$", f"ztarget         = {zt:g}", c, flags=re.M)
    if len(seeds) > 1:
        c = re.sub(r"^\[setup\]", f"[setup]\nSeedCount       = {len(seeds)}",
                   c, flags=re.M)
    open(conf, "w").write(c)
    # CLASS faults on startup often enough to matter, before the seed is used for
    # anything, so a failure says nothing about the realization. A rerun usually
    # clears it.
    tag = (f"seed {seeds[0]}" if len(seeds) == 1
           else f"seeds {seeds[0]}-{seeds[-1]}")
    for attempt in range(NRETRY):
        with open(f"{rundir}/run.log", "w") as log:
            r = subprocess.run([BIN, conf], cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode == 0 and all(os.path.exists(p) for p in outs.values()):
            return outs
        print(f"   {tag}: monofonIC exited {r.returncode}, retrying "
              f"({attempt+1}/{NRETRY})", flush=True)
        # Hundreds of tasks starting at once contend for the CLASS and HyRec data
        # files, and CLASS segfaults on some of those reads. Tasks run in near
        # lockstep, so a retry after a fixed delay collides again; a random wait
        # spreads them out. Measured skip rate rose from 0.5% at 4 concurrent
        # processes to 4.4% at 500.
        time.sleep(random.uniform(1.0, 5.0*(attempt + 1)))
    raise ICFailure(f"monofonIC failed {NRETRY} times for {tag}; see {rundir}/run.log")


def generate_batch(seeds, rundir):
    """Fields for a run of consecutive seeds, one call if possible.

    A batch that will not generate is retried one seed at a time, so a seed that
    fails every time costs itself rather than the whole batch.
    """
    seeds = list(seeds)
    if len(seeds) == 1:
        try:
            return run_ic(seeds, rundir)
        except ICFailure as e:
            print(f"   {e}\n   seed {seeds[0]}: skipped", flush=True)
            SKIPPED.append(seeds[0])
            return {}
    try:
        return run_ic(seeds, rundir)
    except ICFailure as e:
        print(f"   {e}\n   falling back to one seed at a time", flush=True)
    got = {}
    for s in seeds:
        got.update(generate_batch([s], rundir))
    return got


def consecutive_runs(seeds, maxlen):
    """Split seeds into runs of consecutive values, each at most maxlen long.

    monofonIC's SeedCount writes consecutive seeds, so an arbitrary --seed-list
    is batched only where its values happen to run on.
    """
    out, cur = [], []
    for s in seeds:
        if cur and s == cur[-1] + 1 and len(cur) < maxlen:
            cur.append(s)
        else:
            if cur:
                out.append(cur)
            cur = [s]
    if cur:
        out.append(cur)
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
K2 = kk**2; K2[0, 0, 0] = 1.0
KV = [KX, KY, KZ]
kbin = np.array([kk.ravel()[ib == i+1].mean() for i in range(nb)])
nmodes = np.array([(ib == i+1).sum() for i in range(nb)])
binned = lambda P: np.array([P.ravel()[ib == i+1].mean() for i in range(nb)])


def pencil_slice(axis, i, j, margin=0):
    """Index tuple selecting one pencil. Slicing beats building a mask array: at
    N = 128 the 192 masks would be 3.2 GB.

    margin trims that many cells from each of the four long faces. The smoothed
    field at a cell within a smoothing radius of a face is built partly from
    material outside the pencil, so a quantity averaged over the whole pencil is
    not a property of the pencil alone. Trimming a margin of order the smoothing
    radius leaves the cells that are. Nothing is trimmed along the pencil's long
    axis, which spans the periodic box and has no face.

    Returns None when the margin would leave nothing, which happens whenever the
    smoothing radius approaches half the pencil width.
    """
    if 2*margin >= npen:
        return None
    idx = [None, None, None]
    idx[axis] = slice(None)
    rest = [a for a in range(3) if a != axis]
    idx[rest[0]] = slice(i*npen + margin, (i+1)*npen - margin)
    idx[rest[1]] = slice(j*npen + margin, (j+1)*npen - margin)
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
# Margin in cells for each smoothing radius, from --interior-margin in units of R.
MARGIN_CELLS = ([int(np.ceil(ARGS.interior_margin*R*NGRID/LBOX)) for R in ARGS.smooth]
                if ARGS.interior_margin > 0 else [])

# K2 carries a guard at the origin so that 1/K2 is finite, which is harmless for
# delta because a zero-mean field has no DC mode to distort. A mask does have
# one, and it is most of the mask, so anything convolved with a mask uses the
# true k^2 instead.
gauss_true = lambda R: np.exp(-0.5*kk**2*R**2)

# Fraction of each cell's smoothing kernel that fell inside the pencil,
# w(x) = [W_R * M](x), for cells inside the pencil. Convolution commutes with
# translation, so this is the same array for every pencil sharing an axis and is
# built once per (axis, radius) rather than once per pencil per seed.
KWEIGHT = {}
if ARGS.kernel_weight:
    for _a in range(3):
        _sl = pencil_slice(_a, 0, 0)
        _M = np.zeros((NGRID,)*3)
        _M[_sl] = 1.0
        _Mk = np.fft.fftn(_M)
        for _r, _R in enumerate(ARGS.smooth):
            KWEIGHT[_a, _r] = np.real(np.fft.ifftn(_Mk*gauss_true(_R)))[_sl]
    print("   kernel fraction retained inside the region, mean over the region:")
    for _r, _R in enumerate(ARGS.smooth):
        _f = np.mean([KWEIGHT[_a, _r].mean() for _a in range(3)])
        print(f"     R = {_R:5.1f} Mpc/h   {_f:.3f}")

# Pencils carrying the inside/outside source split, spread over the three axes
# rather than taken from the front of the list, which would be one axis only.
SPLIT_PENCILS = ([] if not ARGS.source_split else
                 [int(round(t)) for t in
                  np.linspace(0, len(PENCILS) - 1,
                              min(ARGS.source_split, len(PENCILS)))])

# Separation grid for xi. A periodic box makes the wrapped pairs real pairs, so
# the circular correlation is the right one and no zero-padding is needed; the
# survey case in notes/xi_estimators.tex needs padding because it is not periodic.
if ARGS.xi:
    _ax = np.minimum(np.arange(NGRID), NGRID - np.arange(NGRID))*(LBOX/NGRID)
    RX, RY, RZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
    rr = np.sqrt(RX**2 + RY**2 + RZ**2)
    r_edges = np.linspace(0.0, ARGS.rmax, ARGS.nrbin + 1)
    r_idx = np.digitize(rr.ravel(), r_edges) - 1
    r_in = (r_idx >= 0) & (r_idx < ARGS.nrbin)
    r_idx = r_idx[r_in]
    r_cnt = np.bincount(r_idx, minlength=ARGS.nrbin)[:ARGS.nrbin]
    rbin = (np.bincount(r_idx, weights=rr.ravel()[r_in], minlength=ARGS.nrbin)[:ARGS.nrbin]
            / np.maximum(r_cnt, 1))

    def xi_bin(num, den=None):
        """Bin a pair sum by separation, dividing by the pair count if given."""
        n = np.bincount(r_idx, weights=num.ravel()[r_in], minlength=ARGS.nrbin)[:ARGS.nrbin]
        if den is None:
            return n/np.maximum(r_cnt, 1)/NGRID**3
        dsum = np.bincount(r_idx, weights=den.ravel()[r_in],
                           minlength=ARGS.nrbin)[:ARGS.nrbin]
        return np.where(dsum > 0, n/np.maximum(dsum, 1e-30), np.nan)

# Pair count of the mask, one per orientation: the mask's shape does not depend
# on which tile it is, only on which axis is the long one.
if ARGS.xi:
    RR_AXIS = {}
    for _a in sorted({p[0] for p in PENCILS}):
        _W = np.zeros((NGRID,)*3); _W[pencil_slice(_a, 0, 0)] = 1.0
        RR_AXIS[_a] = np.real(np.fft.ifftn(np.abs(np.fft.fftn(_W))**2))

W0 = np.zeros((NGRID,)*3); W0[pencil_slice(*PENCILS[0])] = 1.0
fvol = W0.mean()

# ---- theory and the window-convolved theory ---------------------------------
# The reference table was written by a back-scaled run, so it is the right theory
# only for runs that back-scale as well. --ztarget changes which redshift the
# transfer functions are taken at, and the theory has to move with it or the
# comparison is between two different fields.
th = np.loadtxt(ARGS.powerspec or paths.REF_POWERSPEC)
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
ACC = {k: [] for k in ("seed", "P_full", "P_pencil", "xi_full", "xi_pencil",
                       "shear", "dbar", "lambda",
                       "webtype", "contrast", "bulk",
                       "shear_interior", "dbar_interior", "lambda_interior",
                       "shear_box", "dbar_box", "lambda_box", "webtype_box",
                       "shear_kw", "dbar_kw", "webtype_kw",
                       "shear_src", "dbar_src")}
SKIPPED = []


def write_chunk():
    """Write everything accumulated so far as one finished file, and clear it.

    Each call produces a complete, self-contained file that is never reopened,
    which is what makes it safe to read while the sweep is still running. The
    alternative, holding one file open and extending its datasets, would put a
    reader in contention with the writer's lock and let it see half-written
    arrays. It also means a task killed part way through keeps the batches it
    has already written instead of losing all of its seeds.
    """
    if not ACC["seed"]:
        return
    tag = f"{ACC['seed'][0]:06d}_{len(ACC['seed']):05d}"
    with h5py.File(f"{OUT}/chunk_{tag}.hdf5", "w") as f:
        f["k"] = kbin
        # Modes per band, so a reader can tell an excursion from a discrepancy:
        # a Gaussian field scatters by sqrt(2/nmodes) about its mean, and the
        # lowest bands hold only a handful of modes.
        f["nmodes"] = nmodes
        for key, v in ACC.items():
            if v:
                f[key] = np.array(v)
        if ARGS.xi:
            f["r"] = rbin
        f["pencil_axis"] = np.array([p[0] for p in PENCILS])
        f["pencil_i"] = np.array([p[1] for p in PENCILS])
        f["pencil_j"] = np.array([p[2] for p in PENCILS])
        f["P_theory"] = P_theory      # self-contained: no shared file needed
        f["P_win"] = P_win
        if ARGS.environment:
            f["smooth_R"] = np.array(ARGS.smooth)
            if MARGIN_CELLS:
                f["margin_cells"] = np.array(MARGIN_CELLS)
                f.attrs["interior_margin"] = ARGS.interior_margin
            if SPLIT_PENCILS:
                f["split_pencils"] = np.array(SPLIT_PENCILS)
        # Skipped seeds are recorded once, with the batch that was open when the
        # skip happened, so the totals over all files still come out right.
        f["skipped"] = np.array(SKIPPED, dtype=np.int64)
        SKIPPED.clear()
        f.attrs["species"] = np.array(SPECIES, dtype=h5py.string_dtype())
        f.attrs.update(dict(N=NGRID, L=LBOX, frac=FRAC, fvol=fvol, kny=kny,
                            dkperp=dkperp, dofixing=ARGS.dofixing,
                            nseeds=len(ACC["seed"]), seed0=ACC["seed"][0]))
    print(f"wrote {OUT}/chunk_{tag}.hdf5", flush=True)
    for v in ACC.values():
        v.clear()
READY = {}          # seed -> field path, for the batch currently generated
for s in SEEDS:
    t0 = time.time()
    # Unique per chunk: parallel array tasks share one --out directory, so a
    # single _work would have them deleting each other's field mid-run.
    rundir = f"{OUT}/_work_{SEEDS[0]:07d}" if ARGS.compact else f"{OUT}/seed_{s:05d}"
    os.makedirs(rundir, exist_ok=True)
    if s not in READY:
        if ARGS.compact:
            for stale in glob.glob(f"{rundir}/deltaq*.hdf5"):
                os.remove(stale)
        rest = SEEDS[SEEDS.index(s):]
        batch = consecutive_runs(rest, max(1, ARGS.batch_seeds))[0]
        t_gen = time.time()
        READY = generate_batch(batch, rundir)
        if len(batch) > 1:
            print(f"   generated {len(READY)}/{len(batch)} seeds "
                  f"{batch[0]}-{batch[-1]} in {time.time()-t_gen:.1f} s", flush=True)
    fic = READY.pop(s, None)
    if fic is None:
        continue
    with h5py.File(fic) as f:
        d = {sp: f[DSET[sp]][:].astype(float) for sp in SPECIES}
        zstart = float(f["Header"].attrs["zstart"])
        Dplus = float(f["Header"].attrs["Dplus"])

    P_full = np.empty((len(SPECIES), nb))
    P_pen = np.empty((len(SPECIES), len(PENCILS), nb))
    if ARGS.xi:
        xi_full = np.empty((len(SPECIES), ARGS.nrbin))
        xi_pen = np.empty((len(SPECIES), len(PENCILS), ARGS.nrbin))
    for m, sp in enumerate(SPECIES):
        F2 = np.abs(np.fft.fftn(d[sp]))**2
        P_full[m] = binned(V*F2/NGRID**6)
        if ARGS.xi:
            # ifftn(|F|^2)[r] is the sum over x of delta(x) delta(x+r); dividing
            # by the number of pairs, N^3 for the unmasked box, gives xi(r).
            xi_full[m] = xi_bin(np.real(np.fft.ifftn(F2)))
        for n, p in enumerate(PENCILS):
            G2 = np.abs(np.fft.fftn(masked(d[sp], p)))**2
            P_pen[m, n] = binned(V*G2/NGRID**6/fvol)
            if ARGS.xi:
                # Same sum, but only over pairs with both points inside the
                # pencil, divided by the count of those pairs. The mask cancels.
                xi_pen[m, n] = xi_bin(np.real(np.fft.ifftn(G2)), RR_AXIS[p[0]])

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
        # The same quantities for the whole periodic box. They cost a few array
        # reductions, since s2, dsm, ev and npos are already computed everywhere,
        # and they answer a different question: whether selecting a pencil also
        # makes the box it came from atypical, or only the region inside it.
        shear_box = np.empty(nR)
        dbar_box = np.empty(nR)
        lam_box = np.empty((nR, 3))
        web_box = np.empty((nR, 4))
        shear_in = np.full((nR, len(PENCILS)), np.nan)
        dbar_in = np.full((nR, len(PENCILS)), np.nan)
        lam_in = np.full((nR, len(PENCILS), 3), np.nan)
        # Kernel-fraction weighted, and the inside/outside source split. Both
        # ask what part of a region's number belongs to the region; the first
        # reweights the cells, the second decomposes the field that made them.
        shear_kw = np.full((nR, len(PENCILS)), np.nan)
        dbar_kw = np.full((nR, len(PENCILS)), np.nan)
        web_kw = np.full((nR, len(PENCILS), 4), np.nan)
        shear_src = np.full((nR, len(SPLIT_PENCILS), 3), np.nan)  # in, out, cross
        dbar_src = np.full((nR, len(SPLIT_PENCILS), 2), np.nan)   # in, out
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
            dks = dk0*gauss_true(R)
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
            TT2 = (T*T).sum((-1, -2))
            s2 = TT2 - dsm*dsm/3.0
            if s == SEEDS[0] and r == 0:
                print(f"   check <s^2>/<delta^2> = {s2.mean()/dsm.var():.4f} (2/3 expected)")
            # lam[..., 0] is the largest eigenvalue, the axis along which the
            # region collapses first.
            ev = eigvals_sym3(T)
            npos = (ev > 0.0).sum(-1)             # T-web class: 3 knot ... 0 void
            shear_box[r] = np.sqrt(s2.mean())
            dbar_box[r] = dsm.mean()              # zero by construction; a check
            lam_box[r] = ev.mean((0, 1, 2))
            cnt_box = np.bincount(npos.ravel(), minlength=4)
            web_box[r] = cnt_box[::-1]/cnt_box.sum()
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
                # The same quantities on the pencil with a margin trimmed off each
                # long face, so that no cell entering the average was smoothed with
                # material from outside the pencil. NaN where the margin would
                # consume the region, which happens once R approaches half its width.
                if MARGIN_CELLS:
                    sli = pencil_slice(*p, margin=MARGIN_CELLS[r])
                    if sli is None:
                        shear_in[r, n] = dbar_in[r, n] = np.nan
                        lam_in[r, n] = np.nan
                    else:
                        shear_in[r, n] = np.sqrt(s2[sli].mean())
                        dbar_in[r, n] = dsm[sli].mean()
                        lam_in[r, n] = ev[sli].mean((0, 1, 2))
                # The same averages with each cell weighted by how much of its
                # kernel stayed inside: full weight at the centre, about half at
                # a face. Unlike a margin this keeps every cell and stays defined
                # at radii where a margin would leave nothing.
                if KWEIGHT:
                    w = KWEIGHT[ax, r]
                    wsum = w.sum()
                    shear_kw[r, n] = np.sqrt((w*s2[sl]).sum()/wsum)
                    dbar_kw[r, n] = (w*dsm[sl]).sum()/wsum
                    for cls in range(4):
                        web_kw[r, n, 3-cls] = w[npos[sl] == cls].sum()/wsum

            # Where the region's tidal field comes from. Poisson is linear, so
            # splitting the source at the region boundary splits the tensor,
            # T[delta] = T[M delta] + T[(1-M) delta], with no approximation. The
            # outside term is not an error to be removed: the tidal field at a
            # point genuinely responds to distant matter. It is the size of that
            # response, which is what decides whether a number measured on the
            # region describes the region.
            for q, ip in enumerate(SPLIT_PENCILS):
                p = PENCILS[ip]
                sl = pencil_slice(*p)
                dks_in = np.fft.fftn(masked(d["matter"], p))*gauss_true(R)
                d_in = np.real(np.fft.ifftn(dks_in))
                d_out = dsm - d_in
                A = np.zeros(dsm.shape)     # sum_ij T_in,ij^2
                B = np.zeros(dsm.shape)     # sum_ij T_ij T_in,ij
                for a in range(3):
                    for b in range(3):
                        t_in = np.real(np.fft.ifftn(KV[a]*KV[b]/K2*dks_in))
                        A += t_in*t_in
                        B += T[..., a, b]*t_in
                # T_out = T - T_in, so its square and the cross term follow from
                # A, B and sum_ij T_ij^2 without building a third tensor.
                s2_in = A - d_in*d_in/3.0
                s2_out = (TT2 - 2*B + A) - d_out*d_out/3.0
                s2_x = 2*((B - A) - d_in*d_out/3.0)
                shear_src[r, q] = [s2_in[sl].mean(), s2_out[sl].mean(),
                                   s2_x[sl].mean()]
                dbar_src[r, q] = [d_in[sl].mean(), d_out[sl].mean()]
                if s == SEEDS[0] and r == 0 and q == 0:
                    tot = shear_src[r, q].sum()
                    print(f"   check source split closes to "
                          f"{tot/s2[sl].mean():.6f} of <s^2> (1 expected)")

    if ARGS.compact:
        ACC["seed"].append(s)
        ACC["P_full"].append(P_full)
        ACC["P_pencil"].append(P_pen)
        if ARGS.xi:
            ACC["xi_full"].append(xi_full)
            ACC["xi_pencil"].append(xi_pen)
        if ARGS.environment:
            ACC["shear"].append(shear); ACC["dbar"].append(dbar)
            ACC["lambda"].append(lam); ACC["webtype"].append(web)
            ACC["contrast"].append(contrast); ACC["bulk"].append(bulk)
            ACC["shear_box"].append(shear_box); ACC["dbar_box"].append(dbar_box)
            ACC["lambda_box"].append(lam_box); ACC["webtype_box"].append(web_box)
            if MARGIN_CELLS:
                ACC["shear_interior"].append(shear_in)
                ACC["dbar_interior"].append(dbar_in)
                ACC["lambda_interior"].append(lam_in)
            if KWEIGHT:
                ACC["shear_kw"].append(shear_kw)
                ACC["dbar_kw"].append(dbar_kw)
                ACC["webtype_kw"].append(web_kw)
            if SPLIT_PENCILS:
                ACC["shear_src"].append(shear_src)
                ACC["dbar_src"].append(dbar_src)
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
            f["shear_box"] = shear_box
            f["dbar_box"] = dbar_box
            f["lambda_box"] = lam_box
            f["webtype_box"] = web_box
            if MARGIN_CELLS:
                f["shear_interior"] = shear_in
                f["dbar_interior"] = dbar_in
                f["lambda_interior"] = lam_in
            if KWEIGHT:
                f["shear_kw"] = shear_kw
                f["dbar_kw"] = dbar_kw
                f["webtype_kw"] = web_kw
            if SPLIT_PENCILS:
                f["shear_src"] = shear_src   # <s^2> as inside, outside, cross
                f["dbar_src"] = dbar_src     # dbar as inside, outside
                f["split_pencils"] = np.array(SPLIT_PENCILS)
            f["smooth_R"] = np.array(ARGS.smooth)
        if ARGS.xi:
            f["r"] = rbin
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
    if ARGS.compact and ARGS.flush_every and len(ACC["seed"]) >= ARGS.flush_every:
        write_chunk()
    print(f"seed {s}: full-box P/theory median {np.median(P_full[0][fit]/P_theory[0][fit]):.4f}, "
          f"mean pencil/P_win {np.median(P_pen[0][:, fit].mean(0)/P_win[0][fit]):.4f}, "
          f"{time.time()-t0:.1f} s")

if ARGS.compact:
    write_chunk()
    # --keep-fields spares the field itself above, so it must spare the directory
    # holding it too, along with the config and the transfer-function table
    # monofonIC wrote beside it. Those are what a figure needs to reproduce a
    # single realization.
    if not ARGS.keep_fields:
        import shutil
        shutil.rmtree(f"{OUT}/_work_{SEEDS[0]:07d}", ignore_errors=True)

if SKIPPED:
    print(f"skipped {len(SKIPPED)} of {len(SEEDS)} seeds: "
          f"{', '.join(str(x) for x in SKIPPED)}")
if not rows:
    raise SystemExit("every seed in this chunk failed; no summary written")

if ARGS.compact:
    # Every task in an array shares --out, and they finish at about the same
    # time, so writing one summary file from all of them collides on the HDF5
    # lock. One task in a few hundred died here after completing all its seeds.
    # The chunk files already hold everything this file would, so in compact
    # mode it is redundant as well as unsafe.
    print(f"wrote {len(SEEDS)} seeds to chunk files under {OUT}", flush=True)
    raise SystemExit(0)

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
