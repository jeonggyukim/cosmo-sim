#!/usr/bin/env python3
"""
make_music_conf.py — Generate a MUSIC2 config file from the template.

Usage:
    python make_music_conf.py -N 256 -z 45 -L 25
    python make_music_conf.py -N 512 -z 100 -L 500 -o my_run.conf

Arguments:
    -N  Particles per side (must be a power of 2, e.g. 64, 128, 256, 512)
    -z  Starting redshift
    -L  Box size in Mpc/h
    -o  Output config filename (default: CV_22_MUSIC_n{N}_z{z}_L{L}.conf)

The IC output filename embedded in the config will be:
    ics_swift_n{N}_z{z}_L{L}.hdf5
"""

import argparse
import math
import os
import sys


def fmt(x):
    """Format a number dropping unnecessary trailing zeros (e.g. 45.0 -> '45')."""
    return f"{x:g}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a MUSIC2 IC config from the CV_22 template."
    )
    parser.add_argument("-N", "--npart", type=int, required=True,
                        help="Particles per side, must be a power of 2 (e.g. 256)")
    parser.add_argument("-z", "--redshift", type=float, required=True,
                        help="Starting redshift (e.g. 45)")
    parser.add_argument("-L", "--boxlength", type=float, required=True,
                        help="Box size in Mpc/h (e.g. 25)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output config filename (auto-named if omitted)")
    args = parser.parse_args()

    # Validate N is a power of 2
    level = math.log2(args.npart)
    if not level.is_integer():
        sys.exit(f"Error: N={args.npart} is not a power of 2")
    level = int(level)

    # Build filename stems
    N_s = fmt(args.npart)
    z_s = fmt(args.redshift)
    L_s = fmt(args.boxlength)

    ic_filename   = f"ics_swift_n{N_s}_z{z_s}_L{L_s}.hdf5"
    conf_filename = args.output or f"CV_22_MUSIC_n{N_s}_z{z_s}_L{L_s}.conf"

    # Load template (sits next to this script)
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "CV_22_MUSIC_template.conf")
    with open(template_path) as f:
        template = f.read()

    # Substitute placeholders (use fmt() so 25.0 renders as "25", not "25.0")
    conf = template.format(
        BOXLENGTH=fmt(args.boxlength),
        ZSTART=fmt(args.redshift),
        LEVEL=level,
        FILENAME=ic_filename,
    )

    with open(conf_filename, "w") as f:
        f.write(conf)

    print(f"Written : {conf_filename}")
    print(f"IC file : music_build/{ic_filename}")
    print(f"Run with: ./music_build/MUSIC {conf_filename}")


if __name__ == "__main__":
    main()
