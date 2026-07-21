#!/usr/bin/env python3
"""
make_monofonic_conf.py — Generate a MUSIC2-monofonIC config from the template.

Usage:
    python make_monofonic_conf.py -N 256 -z 200 -L 1000                 # LPT order 3, fixing on
    python make_monofonic_conf.py -N 256 -z 200 -L 1000 --lpt-order 2   # 2LPT
    python make_monofonic_conf.py -N 256 -z 200 -L 1000 --fixing no     # Gaussian-draw amplitudes
    python make_monofonic_conf.py -N 256 -z 200 -L 1000 --seed 42       # paired-suite seed

Arguments:
    -N  Grid cells (= particles) per side.  Need not be a power of two.
    -z  Starting redshift.
    -L  Box size in Mpc/h.
    --lpt-order  1, 2, or 3 (default: 3).
    --fixing     yes|no  Angulo & Pontzen amplitude fixing (default: yes).
                 When no, '_nofix' is appended to all output stems.
    --seed       NGENIC seed (default: 12345).  A given seed also adds '_s{seed}'
                 to the output stems, matching make_music_conf.py.
    --nthreads   OpenMP threads written into the config (default: 8).
    -o           Output config filename (auto-named if omitted).

The IC output filename embedded in the config is:
    data/ics_swift_n{N}_z{z}_L{L}_mono[_nofix][_s{seed}].hdf5
The '_mono' tag keeps monofonIC outputs from colliding with MUSIC outputs at
the same N, z, L; every downstream step keys off the same stem.
"""

import argparse
import os

DEFAULT_SEED = 12345


def fmt(x):
    """Format a number dropping unnecessary trailing zeros (e.g. 200.0 -> '200')."""
    return f"{x:g}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a MUSIC2-monofonIC IC config from the CV_22 template."
    )
    parser.add_argument("-N", "--gridres", type=int, required=True,
                        help="Grid cells (= particles) per side, e.g. 256")
    parser.add_argument("-z", "--redshift", type=float, required=True,
                        help="Starting redshift, e.g. 200")
    parser.add_argument("-L", "--boxlength", type=float, required=True,
                        help="Box size in Mpc/h, e.g. 1000")
    parser.add_argument("--lpt-order", type=int, choices=[1, 2, 3], default=3,
                        help="LPT order (default: 3)")
    parser.add_argument("--fixing", choices=["yes", "no"], default="yes",
                        help="Angulo & Pontzen amplitude fixing (default: yes)."
                             " When no, '_nofix' is appended to output stems.")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"NGENIC random seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--nthreads", type=int, default=8,
                        help="OpenMP threads written into the config (default: 8).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output config filename (auto-named if omitted)")
    args = parser.parse_args()

    fix = args.fixing  # "yes" or "no"
    fix_suffix = "" if fix == "yes" else "_nofix"
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    seed_suffix = f"_s{args.seed}" if args.seed is not None else ""

    N_s = fmt(args.gridres)
    z_s = fmt(args.redshift)
    L_s = fmt(args.boxlength)

    stem = f"n{N_s}_z{z_s}_L{L_s}_mono{fix_suffix}{seed_suffix}"
    ic_filename = f"data/ics_swift_{stem}.hdf5"

    # Write config to conf/ relative to repo root (three levels up from icpipe/cli/<file>.py).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conf_dir = os.path.join(repo_root, "conf")
    conf_filename = args.output or os.path.join(conf_dir, f"CV_22_monofonIC_{stem}.conf")

    template_path = os.path.join(conf_dir, "CV_22_monofonIC_template.conf")
    with open(template_path) as f:
        template = f.read()

    conf = template.format(
        GRIDRES=args.gridres,
        BOXLENGTH=fmt(args.boxlength),
        ZSTART=fmt(args.redshift),
        LPTORDER=args.lpt_order,
        DOFIXING=fix,
        SEED=seed,
        NTHREADS=args.nthreads,
        FILENAME=ic_filename,
    )

    with open(conf_filename, "w") as f:
        f.write(conf)

    print(f"Written : {conf_filename}")
    print(f"IC file : {ic_filename}")
    print(f"LPT order: {args.lpt_order}   fixing: {fix}   seed: {seed}")
    print(f"Run with: ./monofonic_build/monofonIC {conf_filename}")


if __name__ == "__main__":
    main()
