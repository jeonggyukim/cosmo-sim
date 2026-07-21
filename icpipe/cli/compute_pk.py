#!/usr/bin/env python3
"""compute_pk.py — Estimate matter P(k) from a SWIFT IC HDF5 file.

Thin CLI wrapper around icpipe.ICField.power('delta').

Method (provided by the icpipe package):
  1. Load PartType1/Coordinates from the HDF5.
  2. Assign particles to an Ngrid^3 mesh via interlaced CIC (Sefusatti+ 2016).
  3. FFT, deconvolve the squared CIC window W^2 = prod sinc^4(k/2k_Ny).
  4. Average |delta(k)|^2 in log-spaced k-shells to get P(k).
  5. Optional folding (Jenkins+ 1998) extends to k > k_Nyquist.
  6. Write pk_<stem>.txt; columns are k[h/Mpc], P_raw, P_shot_sub,
     P_nodeconv, sigma_P, nmodes, fold_m.

Usage:
    python compute_pk.py ics_swift_n256_z200_L500.hdf5
    python compute_pk.py ics.hdf5 --ngrid 512 --nkbins 30
    python compute_pk.py ics.hdf5 --fold 4 16        # extend to 16x Nyquist
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from icpipe import ICField
from icpipe.io import write_pk
from icpipe.windows import ASSIGNMENT_ORDER


def _measure_one_level(coords, boxsize_mpc, ngrid, h, interlace, nkbins,
                       *, assignment="pcs", kmin_mpch=None, kmax_mpch=None,
                       shot_noise=True):
    """One-level helper: build a transient ICField from already-loaded
    coords (so we can fold positions in-place for sub-Nyquist levels)."""
    # We bypass the file loader by constructing an empty ICField and
    # patching the relevant attrs.  Cheaper than reloading HDF5.
    obj = ICField.__new__(ICField)
    obj.coords = coords
    obj.velocities = None
    obj.boxsize_mpc = boxsize_mpc
    obj.boxsize_mpch = boxsize_mpc * h
    obj.redshift = np.nan
    obj.N = len(coords)
    obj.h = h
    obj.ngrid = ngrid
    obj.interlace = interlace
    obj.assignment = assignment
    obj.order = ASSIGNMENT_ORDER[assignment]
    res = obj.power("delta", nkbins=nkbins,
                    kmin_mpch=kmin_mpch, kmax_mpch=kmax_mpch,
                    shot_noise=shot_noise)
    return res


def main():
    p = argparse.ArgumentParser(description="Estimate P(k) from a SWIFT IC HDF5 file.")
    p.add_argument("hdf5", help="SWIFT IC HDF5 file")
    p.add_argument("--ngrid", type=int, default=None,
                   help="FFT mesh size (default: N^(1/3))")
    p.add_argument("--nkbins", type=int, default=60,
                   help="Number of log k-bins per folding level (default: 60)")
    p.add_argument("--H0", type=float, default=67.11,
                   help="H0 in km/s/Mpc for h conversion (default: 67.11)")
    p.add_argument("--no-interlace", action="store_true", default=False,
                   help="Disable interlaced assignment (default: on)")
    p.add_argument("--assignment", choices=["ngp", "cic", "tsc", "pcs"], default="pcs",
                   help="Mass-assignment scheme (Sefusatti+ 2016). Higher order = "
                        "less near-Nyquist aliasing; 'pcs' + interlacing is accurate to "
                        "~0.01 per cent at the Nyquist frequency (default: pcs). Use "
                        "'cic' for a faster, lower-accuracy estimate.")
    p.add_argument("--fold", type=int, nargs="+", default=[], metavar="M",
                   help="Folding factors (Jenkins+ 1998). Each M folds positions "
                        "into a sub-box of size L/M, extending k_Nyquist by M and "
                        "reducing Poisson shot noise by M^3.  Provide in increasing "
                        "order. WARNING: only meaningful at z<~1-2.")
    p.add_argument("-o", "--output", default=None,
                   help="Output txt file (default: <data_dir>/pk_<stem>.txt)")
    args = p.parse_args()

    h = args.H0 / 100.0
    interlace = not args.no_interlace

    # --- Build base ICField (loads HDF5, caches metadata) ---
    f = ICField(args.hdf5, ngrid=args.ngrid, interlace=interlace, h=h,
                load_velocities=False, assignment=args.assignment)
    print(f)
    print(f"BoxSize    : {f.boxsize_mpc:.4g} Mpc  =  {f.boxsize_mpch:.4g} Mpc/h")
    print(f"k_Nyquist  : {f.knyq_mpch:.4g} h/Mpc")

    # --- Level m=1 (unfolded) ---
    print("\n--- Level m=1 (unfolded) ---")
    base = f.power("delta", nkbins=args.nkbins, shot_noise=True)
    print(f"P_shot     : {base.P_shot:.4e} (Mpc/h)^3")

    all_k, all_Praw, all_Pss, all_Pnd = [base.k], [base.P_raw], [base.P], [base.P_nodeconv]
    all_err, all_n, all_m = [base.sigma_P], [base.nmodes], [np.ones(len(base.k), dtype=int)]
    knyq_prev = base.knyq_mpch

    # --- Folded levels ---
    if args.fold and f.redshift > 2:
        d_mpc = f.boxsize_mpc / round(f.N ** (1/3))
        k_lat = 2 * np.pi / d_mpc / h
        print(f"\nWARNING: folding at z={f.redshift:.4g} is unreliable.")
        print(f"  Particle spacing d={d_mpc:.3g} Mpc -> k_lattice={k_lat:.3g} h/Mpc")
        print("  At high z, rms displacements << d, so folding aliases the lattice")
        print("  power into the measurement range.  Use folding only at z <~ 1-2.")

    for m in sorted(args.fold):
        sub_box = f.boxsize_mpc / m
        kmin_fold = knyq_prev
        kmax_fold = np.pi * f.ngrid / sub_box / h * 0.9
        print(f"\n--- Level m={m} (sub-box {sub_box:.4g} Mpc) ---")
        print(f"k range    : [{kmin_fold:.4g}, {kmax_fold:.4g}] h/Mpc")
        pos_fold = f.coords % sub_box
        res = _measure_one_level(pos_fold, sub_box, f.ngrid, h, interlace,
                                 args.nkbins, assignment=args.assignment,
                                 kmin_mpch=kmin_fold, kmax_mpch=kmax_fold,
                                 shot_noise=True)
        print(f"P_shot     : {res.P_shot:.4e} (Mpc/h)^3 "
              f"(= P_shot_base / {m}^3)")
        all_k.append(res.k);     all_Praw.append(res.P_raw)
        all_Pss.append(res.P);   all_Pnd.append(res.P_nodeconv)
        all_err.append(res.sigma_P);  all_n.append(res.nmodes)
        all_m.append(np.full(len(res.k), m, dtype=int))
        knyq_prev = res.knyq_mpch

    # --- Output ---
    stem = os.path.basename(args.hdf5).replace(".hdf5", "").replace("ics_swift_", "")
    data_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_txt = args.output or os.path.join(data_dir, f"pk_{stem}.txt")

    fold_str = (f"  fold_factors={[1] + sorted(args.fold)}" if args.fold else "")
    header = [
        f"P(k) from {args.hdf5}",
        f"BoxSize={f.boxsize_mpch:.4g} Mpc/h  N={f.N}  ngrid={f.ngrid}{fold_str}",
        f"P_shot(m=1) = {base.P_shot:.4e} (Mpc/h)^3  (NOT subtracted; shown separately)",
    ]

    write_pk(out_txt,
             k=np.concatenate(all_k),
             P_raw=np.concatenate(all_Praw),
             P_ss=np.concatenate(all_Pss),
             P_nodeconv=np.concatenate(all_Pnd),
             sigma_P=np.concatenate(all_err),
             nmodes=np.concatenate(all_n),
             fold_m=np.concatenate(all_m),
             header_lines=header)
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
