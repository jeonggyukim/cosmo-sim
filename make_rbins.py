#!/usr/bin/env python3
"""
make_rbins.py — Generate a logarithmically spaced bin file for compute_xi.

Bins are in Mpc (matching SWIFT HDF5 coordinate units). The range is chosen
to span from just above the mean interparticle spacing to L/4, where L is
the box size.

Usage:
    # From an IC HDF5 file (reads BoxSize and particle count automatically):
    python make_rbins.py --hdf5 music_build/ics_swift_n256_z127_L25.hdf5

    # Or specify N and L directly (L in Mpc/h, converted to Mpc using H0):
    python make_rbins.py -N 256 -L 25 --H0 67.11

    # Override output file and number of bins:
    python make_rbins.py --hdf5 ics.hdf5 -o mybins.txt --nbins 30

Output format: two-column ASCII (rmin rmax) per line, one bin per row.
"""

import argparse
import math
import sys

import numpy as np


def bins_from_box(boxsize_mpc, npart, nbins):
    """
    Compute logarithmically spaced bin edges.

    Parameters
    ----------
    boxsize_mpc : float
        Box size in Mpc.
    npart : int
        Particles per side (N, so total particles = N^3).
    nbins : int
        Number of bins.

    Returns
    -------
    edges : ndarray, shape (nbins+1,)
        Bin edges in Mpc.
    """
    mean_spacing = boxsize_mpc / npart   # mean interparticle separation in Mpc
    rmin = 2.0 * mean_spacing            # start safely above the mean spacing
    rmax = boxsize_mpc / 3.0            # stay well within periodic boundary artifacts (hard limit is L/2)

    if rmin >= rmax:
        sys.exit(
            f"Error: rmin ({rmin:.3f} Mpc) >= rmax ({rmax:.3f} Mpc). "
            "Box too small or N too coarse for meaningful bins."
        )

    edges = np.logspace(math.log10(rmin), math.log10(rmax), nbins + 1)
    return edges


def read_hdf5_params(hdf5_path):
    """Read BoxSize (Mpc) and particle count from a SWIFT IC HDF5 file."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        boxsize = float(f["Header"].attrs["BoxSize"])
        npart_total = int(f["Header"].attrs["NumPart_Total"][1])  # PartType1
    npart = round(npart_total ** (1.0 / 3.0))
    if npart ** 3 != npart_total:
        sys.exit(f"Error: particle count {npart_total} is not a perfect cube.")
    return boxsize, npart


def main():
    parser = argparse.ArgumentParser(
        description="Generate rbins for compute_xi appropriate for a given IC run."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdf5", metavar="FILE",
                     help="SWIFT IC HDF5 file (reads BoxSize and N automatically)")
    src.add_argument("-L", "--boxlength", type=float,
                     help="Box size in Mpc/h (requires --H0 and -N)")

    parser.add_argument("-N", "--npart", type=int,
                        help="Particles per side (required with -L)")
    parser.add_argument("--H0", type=float, default=67.11,
                        help="Hubble constant in km/s/Mpc (default: 67.11, used to convert Mpc/h → Mpc)")
    parser.add_argument("--nbins", type=int, default=24,
                        help="Number of bins (default: 24)")
    parser.add_argument("-o", "--output", default="rbins.txt",
                        help="Output filename (default: rbins.txt)")
    args = parser.parse_args()

    if args.hdf5:
        boxsize_mpc, npart = read_hdf5_params(args.hdf5)
    else:
        if args.npart is None:
            sys.exit("Error: -N is required when using -L")
        h = args.H0 / 100.0
        boxsize_mpc = args.boxlength / h   # convert Mpc/h → Mpc
        npart = args.npart

    edges = bins_from_box(boxsize_mpc, npart, args.nbins)

    with open(args.output, "w") as f:
        for lo, hi in zip(edges[:-1], edges[1:]):
            f.write(f"{lo:.6f} {hi:.6f}\n")

    mean_spacing = boxsize_mpc / npart
    print(f"BoxSize        : {boxsize_mpc:.4g} Mpc")
    print(f"N per side     : {npart}")
    print(f"Mean spacing   : {mean_spacing:.4g} Mpc")
    print(f"r range        : [{edges[0]:.4g}, {edges[-1]:.4g}] Mpc")
    print(f"Bins           : {args.nbins}")
    print(f"Written        : {args.output}")


if __name__ == "__main__":
    main()
