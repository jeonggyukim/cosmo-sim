#!/usr/bin/env python3
"""Is a single pencil's xi more often below the theory than above it?

The estimator is unbiased in the mean: the mask's pair count divides out, so
<xi_hat> = xi exactly. That says nothing about the median. xi_hat is a sum of
products of correlated Gaussian fields over a small volume, and at separations
where the pencil holds few pairs its distribution is strongly right-skewed. A
skewed positive quantity has its median below its mean, so most pencils can sit
below the truth while a handful of large ones hold the average on it.

The distinction matters for how a figure is read. The mean curve landing on
theory is the estimator behaving correctly; it does not mean a typical zoom-in
region measures the truth. If the median runs low, then most regions selected
this way would report a xi below theory, and that is a statement about what a
practitioner would actually see.

    python check_xi_skew.py --data DIR [--species matter]
"""
import argparse, glob, os
import numpy as np, h5py
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "xismoke"))
ap.add_argument("--species", default="matter")
ap.add_argument("--rmax", type=float, default=60.0,
                help="largest separation counted. Beyond it the linear xi "
                     "approaches its zero crossing, and a ratio to a vanishing "
                     "denominator measures the crossing rather than the estimator")
ap.add_argument("--one-per-seed", action="store_true",
                help="use a single pencil from each realization, so the sample "
                     "carries no two subvolumes that share a box's modes")
A = ap.parse_args()

files = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
if not files:
    raise SystemExit(f"no chunk_*.hdf5 under {A.data}")
XP, XF = [], []
with h5py.File(files[0]) as f:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    i = names.index(A.species)
    r, k = f["r"][:], f["k"][:]
    Pth = f["P_theory"][i]
    L, N = float(f.attrs["L"]), int(f.attrs["N"])
    frac = int(f.attrs["frac"])
    edges = f["r_edges"][:] if "r_edges" in f else np.linspace(0.0, 250.0, len(r) + 1)
for fn in files:
    with h5py.File(fn) as f:
        xp = f["xi_pencil"][:, i]
        XP.append(xp[:, 0] if A.one_per_seed else xp.reshape(-1, xp.shape[-1]))
        XF.append(f["xi_full"][:, i])
XP = np.concatenate(XP); XF = np.concatenate(XF)
width = L/frac

# Linear xi on the measurement's own bins, from the same spectrum the sweep used.
kf = 2*np.pi/L
ka = np.fft.fftfreq(N, d=1.0/N)*kf
k3 = np.sqrt(sum(g**2 for g in np.meshgrid(ka, ka, ka, indexing="ij")))
P3 = np.zeros_like(k3); pos = k3 > 0
P3[pos] = np.exp(np.interp(np.log(k3[pos]), np.log(k), np.log(Pth)))
x3 = np.real(np.fft.ifftn(P3))*N**3/L**3
ra = np.minimum(np.arange(N), N - np.arange(N))*(L/N)
r3 = np.sqrt(sum(g**2 for g in np.meshgrid(ra, ra, ra, indexing="ij")))
idx = np.digitize(r3.ravel(), edges) - 1
inb = (idx >= 0) & (idx < len(r))
cnt = np.bincount(idx[inb], minlength=len(r))[:len(r)]
xt = (np.bincount(idx[inb], weights=x3.ravel()[inb], minlength=len(r))[:len(r)]
      / np.maximum(cnt, 1))

n = len(XP)
print(f"{n} pencils from {len(XF)} realizations"
      f"{', one per realization' if A.one_per_seed else ''}")
print(f"pencil width {width:.1f} Mpc/h, cell {L/N:.2f} Mpc/h\n")
print(f"{'r':>7} {'mean/th':>9} {'median/th':>10} {'below':>7} {'+/-':>6} {'skew':>7}")
rows = []
for j in range(len(r)):
    if not (0 < r[j] <= min(width, A.rmax) and xt[j] > 0):
        continue
    v = XP[:, j]/xt[j]
    below = np.mean(v < 1.0)
    # Binomial error on the fraction. Pencils inside one box share its modes, so
    # with all pencils kept this understates the true uncertainty; --one-per-seed
    # removes that correlation at the cost of a smaller sample.
    err = np.sqrt(below*(1 - below)/n)
    sk = ((v - v.mean())**3).mean()/max(v.std()**3, 1e-30)
    rows.append((r[j], v.mean(), np.median(v), below, err, sk))
    print(f"{r[j]:7.1f} {v.mean():9.4f} {np.median(v):10.4f} {below:7.3f} "
          f"{err:6.3f} {sk:7.2f}")

m = (r > 0) & (r <= min(width, A.rmax)) & (xt > 0)
V = XP[:, m]/xt[m]
below = np.mean(V < 1.0)
print(f"\npooled over every pencil and separation below {min(width, A.rmax):.0f} Mpc/h:")
print(f"  mean            {V.mean():.4f}")
print(f"  median          {np.median(V):.4f}")
print(f"  fraction below  {below:.4f}")
print(f"  median / mean   {np.median(V)/V.mean():.4f}")
if rows:
    fb = np.array([x[3] for x in rows])
    print(f"\nfraction below theory across the {len(rows)} bins: "
          f"min {fb.min():.3f}, median {np.median(fb):.3f}, max {fb.max():.3f}")
    print(f"bins where more than half the pencils fall below theory: "
          f"{int((fb > 0.5).sum())} of {len(rows)}")
