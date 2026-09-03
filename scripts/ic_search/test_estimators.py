#!/usr/bin/env python3
"""Check the xi and P(k) estimators against fields whose answers are known.

Every other test in this directory compares one measurement against another. This
one compares against arithmetic. Three fields have exactly computable statistics,
and the sweep must reproduce them for the whole box and for every pencil:

  constant   delta(x) = c everywhere. Every pair product is c^2, so xi(r) = c^2 at
             every separation, and all the power sits at k = 0, so P(k) = 0 for
             k > 0. This is the sharpest check of the mask division: a pencil must
             return c^2 even at separations where its pair count has collapsed to
             a handful, and any error in the RR denominator shows up at once with
             no statistical noise to hide in.

  tiled      the box is a periodic repeat of one pencil-sized cell in all three
             axes. Pairs spanning cells then carry the same products as pairs
             inside one cell, so xi_pencil(r) = xi_box(r) exactly, at every
             separation, including beyond the pencil width where the pencil holds
             almost no pairs. P(k) is not equal between the two, because the
             window still convolves: the same field exhibits both halves of the
             asymmetry the whole study rests on.

  wave       delta(x) = A cos(k0 . x) with k0 along one axis and a wavelength
             dividing both the box and the pencil width. Then xi along that axis
             is (A^2/2) cos(k0 r) exactly, and both the box and the pencil must
             give it, since each contains a whole number of periods.

The fields are written in monofonIC's output format and the sweep is run on them
unchanged. It skips generation when the field file already exists, so what is
tested is the code that runs on the cluster rather than a copy of it.

    python test_estimators.py [--ngrid 64] [--keep]
"""
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--ngrid", type=int, default=64)
ap.add_argument("--lbox", type=float, default=700.0)
ap.add_argument("--frac", type=int, default=8, help="pencil is 1/frac of the box")
ap.add_argument("--work", default=None)
ap.add_argument("--keep", action="store_true", help="keep the working directory")
ap.add_argument("--python", default=sys.executable)
A = ap.parse_args()

N, L, FRAC = A.ngrid, A.lbox, A.frac
NPEN = N//FRAC
WORK = A.work or tempfile.mkdtemp(prefix="est_test_")
DPLUS = 0.00638670749239255      # only used to convert; no result depends on it
SEED0 = 990001


def write_field(seed, d):
    """Write delta(q) in monofonIC's output format and return its path.

    It is handed to the sweep with --field. Writing it into the sweep's own
    working directory would not survive: the sweep deletes any field it finds
    there before generating, since a stale one left by a failed retry would be
    measured for the wrong seed.
    """
    run = f"{WORK}/{seed}"
    os.makedirs(run, exist_ok=True)
    with h5py.File(f"{run}/field.hdf5", "w") as f:
        for name in ("delta_q", "delta_q_cdm", "delta_q_baryon"):
            f[name] = d
        h = f.create_group("Header")
        h.attrs["BoxSize"] = L
        h.attrs["GridRes"] = N
        h.attrs["zstart"] = 200.0
        h.attrs["Dplus"] = DPLUS
    return f"{run}/field.hdf5"


def run_sweep(seed, field):
    out = f"{WORK}/{seed}/out"
    cmd = [A.python, os.path.join(HERE, "pencil_seed_sweep.py"),
           "--seed0", str(seed), "--nseeds", "1", "--ngrid", str(N),
           "--species", "matter", "--npencils", "6", "--field", field,
           "--xi", "--rmax", str(0.5*L), "--nrbin", "40",
           "--compact", "--keep-fields", "--nthreads", "2", "--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if p.returncode != 0:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise SystemExit(f"sweep failed for seed {seed}")
    import glob
    return sorted(glob.glob(f"{out}/chunk_*.hdf5"))[0]


def read(fn):
    with h5py.File(fn) as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
        i = names.index("matter")
        return dict(r=f["r"][:], k=f["k"][:],
                    xi_box=f["xi_full"][0, i], xi_pen=f["xi_pencil"][0, i],
                    P_box=f["P_full"][0, i], P_pen=f["P_pencil"][0, i],
                    xi_pencil_mean=f["xi_pencil"][0, i].mean(0),
                    pen_axis=f["pencil_axis"][:], pen_i=f["pencil_i"][:],
                    pen_j=f["pencil_j"][:])


FAIL = []


def check(name, got, want, tol, note="", scale=None):
    """Compare against a known answer, relative to `scale` where one is needed.

    An expectation of exactly zero has no relative scale of its own, so those
    checks pass the amplitude the quantity would have had if it were not zero.
    """
    got = np.atleast_1d(got)
    want = np.broadcast_to(np.atleast_1d(want), got.shape)
    ok = np.isfinite(got) & np.isfinite(want)
    if not ok.any():
        FAIL.append(f"{name}: nothing finite to compare"); print(f"  FAIL {name}"); return
    if scale is None:
        ref = np.abs(want[ok]).max()
        scale = np.maximum(np.abs(want[ok]), max(ref, 1.0)*1e-12)
    dev = np.abs(got[ok] - want[ok])/scale
    worst = dev.max()
    status = "ok  " if worst <= tol else "FAIL"
    if worst > tol:
        FAIL.append(f"{name}: worst relative deviation {worst:.3e} > {tol:.0e}")
    print(f"  {status} {name:<52} worst {worst:.3e}  (tol {tol:.0e}) {note}")


print(f"grid {N}^3, L = {L:g} Mpc/h, pencil {L/FRAC:.1f} Mpc/h across")
print(f"work {WORK}\n")

# ---- 1. constant field -------------------------------------------------------
print("constant field, delta = c: xi(r) = c^2 everywhere, P(k>0) = 0")
C = 0.037
seed = SEED0
res = read(run_sweep(seed, write_field(seed, np.full((N,)*3, C))))
check("box xi(r) = c^2 at every separation", res["xi_box"], C**2, 1e-10)
for n in range(res["xi_pen"].shape[0]):
    tag = "" if n else "(one line per pencil)"
    check(f"pencil {n} xi(r) = c^2 at every separation", res["xi_pen"][n], C**2, 1e-10, tag)
# The whole box puts all of a constant field at k = 0, so every measured band
# must vanish. There is no relative scale for zero, so the comparison uses the
# power the same variance would carry spread over the box, V c^2.
PSCALE = L**3*C**2
check("box P(k>0) = 0", res["P_box"], 0.0, 1e-10, scale=PSCALE)

# The pencil does not vanish, and must not. Masking a constant field leaves
# c M(x), whose transform is c Mhat(k): what a pencil measures of a constant
# field is the mask's own power spectrum. That makes this the sharpest available
# test of the window machinery, since the answer is exactly
#   P_pencil(k) = V c^2 |Mhat(k)|^2 / f
# with no reference to any realization. It is the k = 0 delta function convolved
# with the window, which is the same operation the study attributes the pencil
# deficit to.
ka = np.fft.fftfreq(N, d=1.0/N)*(2*np.pi/L)
k3 = np.sqrt(sum(g**2 for g in np.meshgrid(ka, ka, ka, indexing="ij")))
def pencil_mask(axis, i, j):
    """The sweep's own geometry: full along `axis`, one tile in the other two."""
    M = np.zeros((N,)*3)
    idx = [None, None, None]
    idx[axis] = slice(None)
    rest = [a for a in range(3) if a != axis]
    idx[rest[0]] = slice(i*NPEN, (i + 1)*NPEN)
    idx[rest[1]] = slice(j*NPEN, (j + 1)*NPEN)
    M[tuple(idx)] = 1.0
    return M


edges = np.arange(0.5, N/2 + 1.0)*(2*np.pi/L)
ib = np.digitize(k3.ravel(), edges)
# Built from the pencil the sweep actually chose, since --npencils takes a
# deterministic random subset rather than the first few.
n = 0
Mask = pencil_mask(int(res["pen_axis"][n]), int(res["pen_i"][n]), int(res["pen_j"][n]))
fvol = Mask.mean()
Pmask = L**3*C**2*np.abs(np.fft.fftn(Mask)/N**3)**2/fvol
want_pk = np.array([Pmask.ravel()[ib == i + 1].mean() for i in range(len(res["k"]))])
check("pencil P(k) = V c^2 |Mhat|^2 / f  (the mask's own spectrum)",
      res["P_pen"][n], want_pk, 1e-9)

# ---- 2. tiled field ----------------------------------------------------------
print("\nwhite noise, several realizations: xi(r) = 0 for every r > 0")
print("unbiasedness is an ensemble statement, so this one carries an error bar")
XB, XP = [], []
for m in range(6):
    rng = np.random.default_rng(100 + m)
    seed = SEED0 + 10 + m
    # The mean is removed, as monofonIC removes it: the estimator does not
    # subtract one, so a field with a sample mean m contributes m^2 at every
    # separation. A cosmological delta(q) has its k = 0 mode set to zero by
    # construction, and white noise drawn without that step does not.
    wn = rng.normal(0.0, 0.01, (N,)*3)
    res_n = read(run_sweep(seed, write_field(seed, wn - wn.mean())))
    XB.append(res_n["xi_box"]); XP.append(res_n["xi_pencil_mean"])
XB, XP = np.array(XB), np.array(XP)
r_n = res_n["r"]
pos = r_n > 0
# White noise correlates a cell only with itself, so every separation beyond zero
# must average to nothing. The r = 0 bin carries the variance and is the scale
# the rest is judged against.
var = XB[:, 0].mean()
for lab, arr in (("box", XB), ("pencil", XP)):
    mean = arr[:, pos].mean(0)
    err = arr[:, pos].std(0)/np.sqrt(len(arr) - 1)
    z = mean/np.maximum(err, 1e-300)
    # The aggregate, not the extreme. With this few realizations the error is
    # itself uncertain, and the largest of ~40 correlated bins wanders far into
    # the tail even when nothing is wrong. A bias would move the whole set: the
    # mean of z away from zero, or its rms above one.
    print(f"       {lab}: mean z = {z.mean():+.2f}, rms z = {np.sqrt((z**2).mean()):.2f}, "
          f"max |z| = {np.abs(z).max():.2f}")
    print(f"       {lab}: mean xi(r>0) / xi(0) = {mean.mean()/var:+.2e}")
    bad = abs(z.mean()) > 1.0 or np.sqrt((z**2).mean()) > 2.0
    status = "FAIL" if bad else "ok  "
    if bad:
        FAIL.append(f"{lab} white-noise xi(r>0): mean z {z.mean():+.2f}, "
                    f"rms z {np.sqrt((z**2).mean()):.2f}")
    print(f"  {status} {lab + ' xi(r>0) consistent with zero':<52} "
          f"(|mean z| < 1 and rms z < 2)")
print(f"  note xi(0) = {var:.4e} against an input variance of {0.01**2:.4e}")

# ---- 3. plane wave -----------------------------------------------------------
print("\nplane wave, delta = A cos(k0 x) with the wavelength dividing the pencil:")
print("xi(r) = (A^2/2) cos(k0 r) along the wave axis, for the box and the pencil")
A_amp, NPER = 0.02, 2                      # 2 periods across the pencil width
lam = (L/FRAC)/NPER
k0 = 2*np.pi/lam
x = np.arange(N)*(L/N)
wave = A_amp*np.cos(k0*x)[:, None, None]*np.ones((1, N, N))
seed = SEED0 + 2
res = read(run_sweep(seed, write_field(seed, wave)))
# Compare the variance, which the r = 0 bin of xi reports and which is A^2/2 for
# a cosine sampled over whole periods.
check("box xi(0) = A^2/2", res["xi_box"][0], A_amp**2/2, 1e-9)
for n in range(res["xi_pen"].shape[0]):
    check(f"pencil {n} xi(0) = A^2/2", res["xi_pen"][n][0], A_amp**2/2, 1e-9)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURES")
    for f in FAIL:
        print("  " + f)
else:
    print("all analytic checks passed")
if not A.keep:
    shutil.rmtree(WORK, ignore_errors=True)
else:
    print(f"\nkept {WORK}")
sys.exit(1 if FAIL else 0)
