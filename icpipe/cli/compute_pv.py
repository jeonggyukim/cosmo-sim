#!/usr/bin/env python3
"""compute_pv.py — Estimate the velocity power spectrum P_v(k) from a SWIFT IC HDF5.

Thin CLI wrapper around icpipe.ICField.power('velocity').

Method (provided by the icpipe package):
  1. Load PartType1/Coordinates and PartType1/Velocities (v_pec, km/s).
  2. CIC-deposit the momentum field rho*v_a per axis and the density
     rho on the same Ngrid^3 grid; form v_a(x) = momentum_a/rho.
     (Empty cells, rare in IC files, get v=0.)
  3. FFT each component and form the vector spectrum
        P_v(k) = sum_a |v_a(k)|^2.
  4. Deconvolve the squared CIC window W^2 = prod sinc^4(k/2k_Ny).
  5. Average in log-spaced k-shells.
  6. Write pv_<stem>.txt with columns k[h/Mpc], Pv_raw, Pv_nodeconv,
     sigma_Pv, nmodes; Pv units are (km/s)^2 (Mpc/h)^3.

The natural linear-theory prediction is
    P_v(k) = (a*H(z)*f(z))^2 * P_delta(k) / k^2
which can be overlaid with plot_ic.py.

Usage:
    python compute_pv.py ics_swift_n256_z200_L500.hdf5
    python compute_pv.py ics.hdf5 --ngrid 512 --nkbins 30
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from icpipe import ICField
from icpipe.io import write_pv


def main():
    p = argparse.ArgumentParser(description="Estimate P_v(k) from a SWIFT IC HDF5 file.")
    p.add_argument("hdf5", help="SWIFT IC HDF5 file")
    p.add_argument("--ngrid", type=int, default=None,
                   help="FFT mesh size (default: N^(1/3))")
    p.add_argument("--nkbins", type=int, default=60,
                   help="Number of log k-bins (default: 60)")
    p.add_argument("--H0", type=float, default=67.11,
                   help="H0 in km/s/Mpc for h conversion (default: 67.11)")
    p.add_argument("--no-interlace", action="store_true", default=False,
                   help="Disable interlaced CIC (default: on)")
    p.add_argument("-o", "--output", default=None,
                   help="Output txt file (default: <data_dir>/pv_<stem>.txt)")
    args = p.parse_args()

    h = args.H0 / 100.0
    interlace = not args.no_interlace

    f = ICField(args.hdf5, ngrid=args.ngrid, interlace=interlace, h=h,
                load_velocities=True)
    print(f)
    print(f"BoxSize    : {f.boxsize_mpc:.4g} Mpc  =  {f.boxsize_mpch:.4g} Mpc/h")
    print(f"k_Nyquist  : {f.knyq_mpch:.4g} h/Mpc")

    print("\n--- Computing P_v(k) ---")
    res = f.power("velocity", nkbins=args.nkbins)
    print(f"Bins       : {len(res.k)}")
    print(f"P_v(k=k_min)  = {res.P_raw[0]:.4e} {res.units_P}")
    print(f"P_v(k=k_max)  = {res.P_raw[-1]:.4e} {res.units_P}")

    # --- Output ---
    stem = os.path.basename(args.hdf5).replace(".hdf5", "").replace("ics_swift_", "")
    data_dir = os.path.dirname(os.path.abspath(args.hdf5))
    out_txt = args.output or os.path.join(data_dir, f"pv_{stem}.txt")

    header = [
        f"P_v(k) from {args.hdf5}  (vector spectrum, sum over 3 components)",
        f"BoxSize={f.boxsize_mpch:.4g} Mpc/h  N={f.N}  ngrid={f.ngrid}",
        f"z={f.redshift:.4g}; velocities are peculiar (km/s).",
        f"Theory: P_v(k) = (a*H(z)*f(z))^2 * P_delta(k) / k^2.",
    ]
    write_pv(out_txt,
             k=res.k, Pv_raw=res.P_raw, Pv_nodeconv=res.P_nodeconv,
             sigma_Pv=res.sigma_P, nmodes=res.nmodes,
             header_lines=header)
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
