"""Analyse the saved pencil sweep over arbitrary k bands.

Reads OUT/seed_*/pk.hdf5 and OUT/theory.hdf5 written by pencil_seed_sweep.py and
reports, per band, how close individual pencils come to the raw theory and to the
window-convolved theory. The band matters more than anything else: the window
deficit is 34% at the fundamental and under 1% above 2 dk_perp, so "matches
theory" is a statement about which k are included.
"""
import glob, numpy as np, h5py

import argparse, os
import paths

_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default=os.path.join(paths.DATA, "pencil_sweep_n64_L700"))
_ap.add_argument("--species", default="matter", choices=["matter", "cdm", "baryon"])
ARGS = _ap.parse_args()
OUT = ARGS.data
with h5py.File(f"{OUT}/theory.hdf5") as f:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["species"]]
    SP = names.index(ARGS.species)
    k, P_th, P_win = f["k"][:], f["P_theory"][SP], f["P_win"][SP]
    kny, dkperp = f.attrs["kny"], f.attrs["dkperp"]

seeds, P_pen, P_full, tags = [], [], [], []
for fn in sorted(glob.glob(f"{OUT}/seed_*/pk.hdf5")):
    with h5py.File(fn) as f:
        P_pen.append(f["P_pencil"][SP])
        P_full.append(f["P_full"][SP])
        seeds.append(int(f.attrs["seed"]))
        tags.append(np.stack([f["pencil_axis"][:], f["pencil_i"][:], f["pencil_j"][:]], 1))
P_pen = np.concatenate(P_pen)                       # (nseed*npencil, nbins)
seed_of = np.repeat(seeds, len(tags[0]))
tags = np.concatenate(tags)
P_full = np.array(P_full)

print(f"{len(seeds)} seeds x {len(tags)//len(seeds)} pencils = {len(P_pen)} pencils, "
      f"{len(k)} k bins, species = {ARGS.species}")
print(f"full-box P/theory: min {(P_full/P_th).min():.5f}, max {(P_full/P_th).max():.5f}; "
      f"across-seed spread {np.abs(P_full/P_full.mean(0) - 1).max():.1e} "
      f"(zero to machine precision when DoFixing = yes: the seed sets phases, not amplitudes)\n")

BANDS = {
    "all k (k_f .. 0.9 k_Ny)":      (k > 0)            & (k <= 0.9*kny),
    "low k (k <= 2 dk_perp)":       (k <= 2*dkperp),
    "high k (k > 2 dk_perp)":       (k > 2*dkperp)     & (k <= 0.9*kny),
}
for name, sel in BANDS.items():
    dth = np.sqrt((np.log(P_pen[:, sel]/P_th[sel])**2).mean(1))
    dwin = np.sqrt((np.log(P_pen[:, sel]/P_win[sel])**2).mean(1))
    wdef = (P_win[sel]/P_th[sel])
    i = dth.argmin()
    print(f"--- {name}: {sel.sum()} bins, k = {k[sel][0]:.4f}..{k[sel][-1]:.4f}, "
          f"window deficit P_win/P_th = {wdef.min():.3f}..{wdef.max():.3f}")
    print(f"    rms ln(P/P_theory) : min {dth.min():.4f}  median {np.median(dth):.4f}")
    print(f"    rms ln(P/P_win)    : min {dwin.min():.4f}  median {np.median(dwin):.4f}")
    print(f"    best vs raw theory : seed {seed_of[i]} axis {tags[i,0]} (i,j)=({tags[i,1]},{tags[i,2]}) "
          f"rms {100*dth[i]:.1f}%")
    for x in (0.02, 0.05, 0.10):
        print(f"    within {100*x:4.0f}% of raw theory : {(dth < x).sum():5d}/{len(dth)} "
              f"({100*(dth < x).mean():6.2f}%)  ->  ~{(1/max((dth < x).mean(), 1e-9)):.0f} pencils per hit")
    print()
