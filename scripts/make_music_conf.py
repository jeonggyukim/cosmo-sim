#!/usr/bin/env python3
"""
make_music_conf.py — Generate a MUSIC2 config file from the template.

Usage:
    python make_music_conf.py -N 256 -z 127 -L 25              # canonical CV_22 run
    python make_music_conf.py -N 512 -z 127 -L 500 -o my.conf  # custom output
    python make_music_conf.py -N 256 -z 127 -L 25 --fix-amplitude no  # random amplitudes

Arguments:
    -N  Particles per side (must be a power of 2, e.g. 64, 128, 256, 512)
    -z  Starting redshift
    -L  Box size in Mpc/h
    -o  Output config filename (default: CV_22_MUSIC_n{N}_z{z}_L{L}[_nofix].conf)
    --fix-amplitude  yes|no  Fix Fourier mode amplitudes to sqrt(P(k)) (default: yes).
                             When no, '_nofix' is appended to all output stems.

The IC output filename embedded in the config will be:
    ics_swift_n{N}_z{z}_L{L}[_nofix].hdf5
"""

import argparse
import math
import os
import re
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
    parser.add_argument("--fix-amplitude", choices=["yes", "no"], default="yes",
                        help="Fix Fourier mode amplitudes to sqrt(P(k)) (default: yes)."
                             " When no, '_nofix' is appended to output stems.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed for the IC level (e.g. 12345).")
    args = parser.parse_args()

    # Validate N is a power of 2
    level = math.log2(args.npart)
    if not level.is_integer():
        sys.exit(f"Error: N={args.npart} is not a power of 2")
    level = int(level)

    fix_amp = args.fix_amplitude  # "yes" or "no"
    suffix = "" if fix_amp == "yes" else "_nofix"
    seed_suffix = f"_s{args.seed}" if args.seed is not None else ""

    # Build filename stems
    N_s = fmt(args.npart)
    z_s = fmt(args.redshift)
    L_s = fmt(args.boxlength)

    ic_filename = f"data/ics_swift_n{N_s}_z{z_s}_L{L_s}{suffix}{seed_suffix}.hdf5"
    # Write config to conf/ relative to repo root (two levels up from scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conf_dir = os.path.join(repo_root, "conf")
    conf_filename = args.output or os.path.join(
        conf_dir, f"CV_22_MUSIC_n{N_s}_z{z_s}_L{L_s}{suffix}{seed_suffix}.conf"
    )

    # Load template from conf/
    template_path = os.path.join(conf_dir, "CV_22_MUSIC_template.conf")
    with open(template_path) as f:
        template = f.read()

    # Substitute placeholders (use fmt() so 25.0 renders as "25", not "25.0")
    conf = template.format(
        BOXLENGTH=fmt(args.boxlength),
        ZSTART=fmt(args.redshift),
        LEVEL=level,
        LEVEL_PAD=f"{level:04d}",
        FILENAME=ic_filename,
    )

    # Override fix_mode_amplitude if requested
    if fix_amp == "no":
        conf = conf.replace("fix_mode_amplitude = yes", "fix_mode_amplitude = no")

    # Override the random seed for this level if requested
    if args.seed is not None:
        conf = re.sub(rf'seed\[{level}\]\s*=\s*\d+',
                      f'seed[{level}]           = {args.seed}', conf)

    with open(conf_filename, "w") as f:
        f.write(conf)

    print(f"Written : {conf_filename}")
    print(f"IC file : {ic_filename}")
    print(f"fix_mode_amplitude: {fix_amp}")
    print(f"Run with: ./music_build/MUSIC {conf_filename}")


if __name__ == "__main__":
    main()
