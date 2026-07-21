# `tools/` — build & ops scripts

Setup, build, cleanup, and SLURM templates.  These are NOT part of
`run_pipeline.py` invocation surface (well — `build-music.sh` and
`build-corrfunc.sh` ARE invoked by `run_pipeline.py` as needed; see
each script's header for the exact resolution order).

## Scripts

### `build-music.sh`

Build MUSIC2 from source into `music_build/`.  Source directory
resolution order:

1. `$MUSIC2_SOURCE_DIR` environment variable, if set.
2. `../MUSIC2` (sibling of the repo root) — default.
3. Clone from `https://github.com/cosmo-sims/MUSIC2` into option 2.

Platform handling:
- **macOS** (`uname -s == Darwin`): sets `FC=gfortran-14`, expects
  Homebrew `fftw`, `gsl`, `hdf5`, `open-mpi`.
- **Cluster** (non-Darwin): `module load gnu12 openmpi4 fftw hdf5 gsl cmake`.

Skips compilation if `music_build/MUSIC` already exists.

```bash
tools/build-music.sh                            # auto-detect or clone
MUSIC2_SOURCE_DIR=/path/to/MUSIC2 tools/build-music.sh
```

### `build-corrfunc.sh`

Clone + build Corrfunc.  Default location:

- **macOS, user jgkim**: `$HOME/Dropbox/Projects/Corrfunc`
- **other**: `$HOME/Corrfunc`

Skips compilation if the static library is already present.

```bash
tools/build-corrfunc.sh
```

### `clean.sh`

Remove generated pipeline outputs.  Two modes:

```bash
tools/clean.sh           # fast: plots, P(k)/ξ tables, rbins, configs, CLASS outputs, compiled binaries
tools/clean.sh --all     # also removes ICs (HDF5), wnoise binaries, and music_build/ (slow to regenerate)
```

### `mpirun_restart.sbatch`

SLURM job template for resuming an MPI-parallel MUSIC2 run on the
`grammar*` cluster.  Edit `--time`, `-N`, `--ntasks-per-node`,
`--exclude` to match the target queue; the body loads matching
modules (`gnu12`, `hwloc`, `libfabric`, ...).  Not invoked by any
script — submit by hand with `sbatch tools/mpirun_restart.sbatch`.
