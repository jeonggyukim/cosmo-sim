#!/usr/bin/env python3
"""run_pipeline — end-to-end IC generation and validation (MUSIC or monofonIC).

Thin CLI over ``icpipe.pipeline``. Runs from the repo checkout given by
``--root`` (default: the current working directory), which must contain
``tools/``, the build dirs, and the ``conf/`` templates.

Examples:
    run-pipeline                                  # MUSIC (default), from the CWD
    run-pipeline --ic-code monofonic --lpt-order 3
    run-pipeline --ic-code monofonic --launcher "srun"     # multi-node on SLURM
    run-pipeline --ic-code monofonic --mpi-ranks 4         # 4 local MPI ranks
    run-pipeline --root /scratch/cosmo-pipeline --ngrid 512
"""

from __future__ import annotations

import argparse
import os

from icpipe.pipeline import RunConfig, run_pipeline


def main():
    p = argparse.ArgumentParser(
        description="End-to-end IC generation and validation (MUSIC or monofonIC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ic-code", choices=["music", "monofonic"], default="music",
                   help="IC generator")
    p.add_argument("--ngrid", default="256", help="Particles per side (N)")
    p.add_argument("--lbox", default="1000", help="Box size in Mpc/h")
    p.add_argument("--zstart", default="200", help="Starting redshift")
    p.add_argument("--nthreads", default=str(os.cpu_count() or 4),
                   help="OpenMP threads per rank")
    p.add_argument("--fix-amplitude", choices=["yes", "no"], default="yes",
                   help="Fix Fourier mode amplitudes (MUSIC fix_mode_amplitude / monofonic DoFixing)")
    p.add_argument("--lpt-order", choices=["1", "2", "3"], default="3",
                   help="LPT order (monofonic only; MUSIC uses use_2LPT)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for the IC white noise (paired-suite runs)")
    p.add_argument("--ic-source-dir", default=None,
                   help="Path to the IC-code source checkout (else sibling dir or clone)")
    p.add_argument("--corrfunc-dir", default=None, help="Path to a built Corrfunc tree")
    p.add_argument("--launcher", default=None,
                   help='MPI launcher prefix for the IC step, e.g. "srun" or "srun -n 128" '
                        "(inherits the SLURM allocation). The orchestrator itself stays single-process.")
    p.add_argument("--mpi-ranks", type=int, default=1,
                   help="If > 1 and no --launcher, run the IC binary under 'mpirun -np N'")
    p.add_argument("--root", default=os.getcwd(),
                   help="Repo checkout to operate in (holds tools/, build dirs, conf/ templates)")
    p.add_argument("--stop-after-ic", action="store_true",
                   help="Exit after IC generation, skipping CLASS / xi / P(k) / plots")
    args = p.parse_args()

    cfg = RunConfig(
        root=os.path.abspath(args.root),
        ic_code=args.ic_code,
        ngrid=args.ngrid,
        lbox=args.lbox,
        zstart=args.zstart,
        nthreads=args.nthreads,
        fix_amplitude=args.fix_amplitude,
        lpt_order=args.lpt_order,
        seed=args.seed,
        ic_source_dir=args.ic_source_dir,
        corrfunc_dir=args.corrfunc_dir,
        launcher=args.launcher,
        mpi_ranks=args.mpi_ranks,
        stop_after_ic=args.stop_after_ic,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
