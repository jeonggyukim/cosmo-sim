#!/usr/bin/env python3
"""What selecting on the raw-theory P(k) does to a region's xi.

A pencil's P(k) estimator is unbiased for the theory convolved with the pencil
window, which sits well below the raw theory at low k. Asking a pencil to match
the raw theory therefore asks its realization scatter to cancel that deficit,
and the only way is an upward fluctuation of the region's large-scale power.

That prediction can be checked without any reference to a window, because xi
carries none: the mask divides out of it exactly. If the selection is doing what
the argument says, the selected pencils must show excess correlation at large
separations, measured against the same theory every other pencil is measured
against. An excess there cannot be attributed to geometry.

    python check_selected_xi.py --data DIR [--nbest 10] [--species matter]
"""
import argparse, glob, os
import numpy as np, h5py
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(paths.DATA, "xismoke"))
ap.add_argument("--species", default="matter")
ap.add_argument("--nbest", type=int, default=10,
                help="pencils kept, ranked by how well their P(k) matches raw theory")
ap.add_argument("--rmax", type=float, default=120.0)
A = ap.parse_args()

files = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
if not files:
    raise SystemExit(f"no chunk_*.hdf5 under {A.data}")
XP, PP, seeds = [], [], []
with h5py.File(files[0]) as f:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    i = names.index(A.species)
    r, k = f["r"][:], f["k"][:]
    P_th, P_win = f["P_theory"][i], f["P_win"][i]
    L, N = float(f.attrs["L"]), int(f.attrs["N"])
    frac, kny = int(f.attrs["frac"]), float(f.attrs["kny"])
    edges = f["r_edges"][:] if "r_edges" in f else np.linspace(0.0, 250.0, len(r) + 1)
for fn in files:
    with h5py.File(fn) as f:
        xp = f["xi_pencil"][:, i]; pp = f["P_pencil"][:, i]
        XP.append(xp.reshape(-1, xp.shape[-1]))
        PP.append(pp.reshape(-1, pp.shape[-1]))
        sd = f["seed"][:] if "seed" in f else np.zeros(xp.shape[0], int)
        seeds.append(np.repeat(sd, xp.shape[1]))
XP = np.concatenate(XP); PP = np.concatenate(PP); seeds = np.concatenate(seeds)
width = L/frac

kf = 2*np.pi/L
ka = np.fft.fftfreq(N, d=1.0/N)*kf
k3 = np.sqrt(sum(g**2 for g in np.meshgrid(ka, ka, ka, indexing="ij")))
P3 = np.zeros_like(k3); pos = k3 > 0
P3[pos] = np.exp(np.interp(np.log(k3[pos]), np.log(k), np.log(P_th)))
x3 = np.real(np.fft.ifftn(P3))*N**3/L**3
ra = np.minimum(np.arange(N), N - np.arange(N))*(L/N)
r3 = np.sqrt(sum(g**2 for g in np.meshgrid(ra, ra, ra, indexing="ij")))
idx = np.digitize(r3.ravel(), edges) - 1
inb = (idx >= 0) & (idx < len(r))
cnt = np.bincount(idx[inb], minlength=len(r))[:len(r)]
xt = (np.bincount(idx[inb], weights=x3.ravel()[inb], minlength=len(r))[:len(r)]
      / np.maximum(cnt, 1))

band = (k > 0) & (k <= 0.9*kny)
D_raw = np.sqrt((np.log(PP[:, band]/P_th[band])**2).mean(1))
D_win = np.sqrt((np.log(PP[:, band]/P_win[band])**2).mean(1))
best_raw = np.argsort(D_raw)[:A.nbest]
best_win = np.argsort(D_win)[:A.nbest]

rng = np.random.default_rng(0)
useed = np.unique(seeds)


def err_of_mean(sel, j):
    """Spread of the selected mean, resampling the realizations it draws from."""
    s = seeds[sel]
    if len(np.unique(s)) < 2:
        return XP[sel, j].std()/np.sqrt(max(len(sel) - 1, 1))
    draws = [XP[sel[np.isin(s, rng.choice(useed, len(useed)))], j] for _ in range(300)]
    return float(np.std([d.mean() for d in draws if len(d)]))


print(f"{len(XP)} pencils from {len(useed)} realizations; "
      f"pencil width {width:.0f} Mpc/h")
print(f"selecting the {A.nbest} closest to each curve over k <= 0.9 k_Ny\n")
print(f"{'r':>7} {'all':>9} {'best->raw':>11} {'+/-':>7} {'best->window':>13} {'+/-':>7}")
for j in range(len(r)):
    if not (0 < r[j] <= A.rmax and xt[j] > 0):
        continue
    print(f"{r[j]:7.1f} {XP[:, j].mean()/xt[j]:9.4f} "
          f"{XP[best_raw, j].mean()/xt[j]:11.4f} {err_of_mean(best_raw, j)/xt[j]:7.4f} "
          f"{XP[best_win, j].mean()/xt[j]:13.4f} {err_of_mean(best_win, j)/xt[j]:7.4f}")

for lab, sel in (("all pencils", np.arange(len(XP))),
                 ("best match to RAW theory", best_raw),
                 ("best match to theory * window", best_win)):
    m = (r > 20) & (r <= min(width, A.rmax)) & (xt > 0)
    w = np.abs(xt[m])
    v = np.average(XP[sel][:, m].mean(0)/xt[m], weights=w)
    print(f"\n{lab:32s} xi/theory over 20 < r < {min(width, A.rmax):.0f} Mpc/h : {v:.4f}")
