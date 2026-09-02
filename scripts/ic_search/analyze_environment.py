#!/usr/bin/env python3
"""What selecting on the pencil power spectrum does to the state of the region.

Applies the proposed selection criterion -- distance at low k from a reference
curve, set by --reference -- and reports how far the surviving realizations differ,
in quantities the criterion never mentions: the tidal shear of the region, its mean overdensity, the
shape of its tidal ellipsoid, its bulk flow, how it stands against its neighbours,
and its power on scales above and below the window width.

Uncertainties come from a bootstrap over REALIZATIONS, not over pencils. Pencils
within one realization share the modes of the same box, and the three orientations
tile it three times over, so treating them as independent would understate every
error bar. Resampling whole realizations keeps the point estimate from all the data
while respecting that correlation.

Reads either layout written by pencil_seed_sweep.py --environment: a directory per
seed (small local runs) or the chunk files written by --compact (cluster arrays).
Run it both ways. With --reference theory the pencils are asked to match the raw
linear spectrum, which their own mean sits about 34% below at the fundamental, so
only an upward fluctuation can succeed. With --reference window they are asked to
match that same theory convolved with the pencil window, which is what the
estimator has for its mean, so nothing atypical is required. The difference
between the two runs is the bias the choice of curve introduces.

See merge_sweeps.py for combining parallel runs of the first kind.
"""
import argparse, glob, os
import numpy as np, h5py
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "env_n128_L700_x400"))
ap.add_argument("--species", default="matter", choices=["matter", "cdm", "baryon"])
ap.add_argument("--keep", type=float, default=0.05,
                help="fraction of pencils retained by the selection")
ap.add_argument("--reference", default="theory", choices=["theory", "window"],
                help="the curve the selection tries to match: 'theory' is the raw "
                     "linear spectrum, which a pencil measures 34%% below at the "
                     "fundamental for purely geometric reasons; 'window' is that "
                     "theory convolved with the pencil window, which is what the "
                     "estimator actually has for its mean. Selecting on 'window' is "
                     "the control: it asks for nothing atypical")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--one-per-seed", type=int, default=0, metavar="NDRAW",
                help="keep a single randomly chosen pencil from each realization, and "
                     "repeat the draw NDRAW times. Pencils inside one box share its "
                     "modes; taking one removes that correlation instead of modelling "
                     "it with the bootstrap, at the cost of using a fraction of the "
                     "data. The reported spread is over the NDRAW draws")
A = ap.parse_args()


def shape_params(lam):
    """Ellipticity and prolateness of the tidal ellipsoid, from sorted eigenvalues.

    The textbook definitions (Bond & Myers 1996; Sheth, Mo & Tormen 2001) divide by
    the trace delta = l1 + l2 + l3, which passes through zero for a region of
    average density and makes the ratio diverge. Dividing instead by the norm
    L = sqrt(sum l_i^2), which is positive definite, keeps the same ordering and
    the same meaning -- how far from spherical, and prolate versus oblate -- while
    staying finite for every region. Both are dimensionless and bounded.
    """
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    L = np.sqrt((lam**2).sum(-1)) + 1e-30
    return (l1 - l3)/(2*L), (l1 - 2*l2 + l3)/(2*L)


def load(path, species):
    """Return (criterion inputs, columns, seed index) from either sweep layout."""
    chunks = sorted(glob.glob(f"{path}/chunk_*.hdf5"))
    seeds = sorted(glob.glob(f"{path}/seed_*/pk.hdf5"))
    if not chunks and not seeds:
        raise SystemExit(f"no chunk_*.hdf5 or seed_*/pk.hdf5 under {path}")

    with h5py.File(f"{path}/theory.hdf5") as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
        SP = names.index(species)
        k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
        meta = {x: f.attrs[x] for x in ("kny", "dkperp", "N", "L")}
    lo = k <= 2*meta["dkperp"]
    hi = (k > 2*meta["dkperp"]) & (k <= 0.9*meta["kny"])

    crit, cols, seed_of, nseed = [], {}, [], 0

    P_ref = P_th if A.reference == "theory" else P_win

    def add(P, env, RS):
        """One realization: P is (npencil, nk), env holds its per-pencil arrays."""
        crit.append(np.sqrt((np.log(P[:, lo]/P_ref[lo])**2).mean(1)))
        cols.setdefault("large-scale power", []).append((P[:, lo]/P_win[lo]).mean(1))
        cols.setdefault("small-scale power", []).append((P[:, hi]/P_win[hi]).mean(1))
        if env is None:
            return
        for r, R in enumerate(RS):
            cols.setdefault(f"tidal shear R={R:g}", []).append(env["shear"][r])
            cols.setdefault(f"mean overdensity R={R:g}", []).append(env["dbar"][r])
            if "contrast" in env:
                cols.setdefault(f"env contrast R={R:g}", []).append(env["contrast"][r])
            if "lambda" in env:
                lam = env["lambda"][r]
                e, p = shape_params(lam)
                cols.setdefault(f"lambda_1 R={R:g}", []).append(lam[..., 0])
                cols.setdefault(f"ellipticity R={R:g}", []).append(e)
                cols.setdefault(f"prolateness R={R:g}", []).append(p)
            if "webtype" in env:
                for w, wname in enumerate(("knot", "filament", "sheet", "void")):
                    cols.setdefault(f"{wname} fraction R={R:g}", []).append(env["webtype"][r][:, w])
        if "bulk" in env:
            cols.setdefault("bulk flow", []).append(np.linalg.norm(env["bulk"], axis=-1))

    if chunks:
        for fn in chunks:
            with h5py.File(fn) as f:
                RS = f["smooth_R"][:] if "smooth_R" in f else []
                for m in range(f["P_pencil"].shape[0]):
                    env = None
                    if "shear" in f:
                        env = {key: f[key][m] for key in
                               ("shear", "dbar", "contrast", "lambda", "webtype", "bulk")
                               if key in f}
                    add(f["P_pencil"][m, SP], env, RS)
                    seed_of.append(np.full(f["P_pencil"].shape[2], nseed)); nseed += 1
    else:
        for fn in seeds:
            with h5py.File(fn) as f:
                RS = f["smooth_R"][:] if "smooth_R" in f else []
                env = {key: f[key][:] for key in
                       ("shear", "dbar", "contrast", "lambda", "webtype", "bulk")
                       if key in f} or None
                add(f["P_pencil"][SP], env, RS)
                seed_of.append(np.full(f["P_pencil"].shape[1], nseed)); nseed += 1

    return (np.concatenate(crit), {n: np.concatenate(v) for n, v in cols.items()},
            np.concatenate(seed_of), nseed, k, lo, meta)


crit, cols, seed_of, nseed, k, lo, meta = load(A.data, A.species)
npen = len(crit)//nseed

print(f"{nseed} realizations x {npen} pencils = {len(crit)} measurements, "
      f"N = {int(meta['N'])}, L = {meta['L']:g} Mpc/h, species {A.species}")
print(f"selection: the closest {100*A.keep:g}% to "
      f"{'unwindowed theory' if A.reference == 'theory' else 'theory * window'} over "
      f"k <= 2 dk_perp ({k[lo][0]:.3f}-{k[lo][-1]:.3f} h/Mpc)\n")


def shift_in_sd(c, T, idx=None):
    """Mean of the kept minus mean of all, in units of the population scatter."""
    if idx is not None:
        c, T = c[idx], T[idx]
    n = max(1, int(round(A.keep*len(c))))
    keep = np.argpartition(c, n)[:n]  # partial: the order inside does not matter
    return (T[keep].mean() - T.mean())/T.std(), T[keep].std()/T.std()


rng = np.random.default_rng(1)
by_seed = [np.where(seed_of == s)[0] for s in range(nseed)]

if A.one_per_seed:
    idx_by_seed = np.array(by_seed)          # (nseed, npen), same count for every seed
    print(f"one pencil per realization, {A.one_per_seed} independent draws\n")
    print(f"{'quantity':<26} {'shift [sd]':>22} {'all pencils':>13}")
    for name, T in cols.items():
        s = np.empty(A.one_per_seed)
        for d in range(A.one_per_seed):
            pick = idx_by_seed[np.arange(nseed), rng.integers(0, npen, nseed)]
            s[d] = shift_in_sd(crit[pick], T[pick])[0]
        print(f"{name:<26} {s.mean():+8.3f} +/- {s.std():.3f} {shift_in_sd(crit, T)[0]:+12.3f}")
    print(f"\n{nseed} realizations; the spread is over the {A.one_per_seed} draws of which "
          f"pencil to take.\nThese draws are independent, so agreement with the last column "
          f"means the\nbootstrap is treating the within-box correlation correctly.")
    raise SystemExit

print(f"{'quantity':<26} {'shift [sd]':>18} {'scatter kept/all':>18}")
for name, T in cols.items():
    point, ratio = shift_in_sd(crit, T)
    boot = np.empty(A.nboot)
    for b in range(A.nboot):
        pick = rng.integers(0, nseed, nseed)
        idx = np.concatenate([by_seed[s] for s in pick])
        boot[b] = shift_in_sd(crit, T, idx)[0]
    lo68, hi68 = np.percentile(boot, [16, 84])
    print(f"{name:<26} {point:+7.3f} [{lo68:+.3f}, {hi68:+.3f}] {ratio:17.2f}")

print(f"\nBootstrap over {nseed} realizations, {A.nboot} resamples; the interval is 16th-84th "
      f"percentile.\nA shift is only meaningful if the interval excludes zero.")
