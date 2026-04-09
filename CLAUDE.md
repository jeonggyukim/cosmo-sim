# CLAUDE.md — cosmo-pipeline

This repo contains scripts and configuration files for running cosmological simulations, primarily IC generation with MUSIC2 and N-body/hydro runs with SWIFT.

## Directory Structure

```
cosmo-pipeline/
  scripts/   — Python scripts (compute_pk.py, make_rbins.py, make_music_conf.py, read_*.py, ...)
  data/      — IC HDF5 files, CLASS P(k) outputs, rbins files, wnoise binaries, measured P/xi tables
  plots/     — PNG/PDF figures (pk_*.png, xi_*.png, ...)
  conf/      — MUSIC2 configs (CV_22_MUSIC*.conf) and log files
  notes/     — LaTeX write-ups and supporting plot scripts
  music_build/ — compiled MUSIC2 binary (gitignored)
```

Note: `data/*.hdf5` and `data/*.bin` are gitignored (large binary files).

## Key Files

- `prepare-music.sh` — builds MUSIC2 from source and places the binary in `music_build/`
- `conf/CV_22_MUSIC.conf` — canonical MUSIC2 config: 25 Mpc/h box, 256^3 (level 8), z=127, SWIFT output
- `conf/CV_22_MUSIC_template.conf` — template config with `{BOXLENGTH}`, `{ZSTART}`, `{LEVEL}`, `{FILENAME}` placeholders
- `scripts/make_music_conf.py` — generates a config from the template given N, z, L
- `scripts/make_rbins.py` — generates Corrfunc bin file appropriate for a given IC run (rmin=2×mean spacing, rmax=L/3)
- `compute_xi.c` / `compute_xi` — measures ξ(r) from IC particles using Corrfunc (C, compiled via `make`)
- `scripts/compute_pk.py` — estimates P(k) from IC particles via CIC + FFT, compares to CLASS theory
- `data/class_pk.ini` — CLASS input for matter P(k); edit `z_pk` and `root` for each redshift
- `data/class_pk_z45_pk.dat` — CLASS P(k) output at z=45 (CV_22 cosmology); naming: `class_pk_z{z}_pk.dat`
- `scripts/read_wnoise.py` — reads `wnoise_NNNN.bin` white noise binary into a numpy array
- `scripts/read_ics_swift.py` — reads `ics_swift.hdf5` IC file (header + particle data)
- `music_build/MUSIC` — compiled MUSIC2 binary (gitignored)
- `music_build/_deps/class-build/class` — CLASS binary (built by MUSIC2's CMake)

## MUSIC2

See [MUSIC2_CLAUDE.md](MUSIC2_CLAUDE.md) for full details on the MUSIC2 code structure, build instructions, IC generation pipeline, transfer function internals, and file formats.

Source lives at `~/Dropbox/Projects/MUSIC2`.

### Quick usage

```bash
# Build (first time or after source changes)
./prepare-music.sh

# Generate a config (canonical CV_22 run: N=256, z=127, L=25 Mpc/h)
conda run -n cosmo python scripts/make_music_conf.py -N 256 -z 127 -L 25

# Run (config is written to conf/ by make_music_conf.py)
./music_build/MUSIC conf/CV_22_MUSIC_n256_z127_L25.conf
```

## Python environment

Use `conda run -n cosmo python ...` for all Python scripts in this project.

## IC validation

SWIFT stores coordinates and BoxSize in **Mpc** (not Mpc/h). All bin files and P(k) calculations use Mpc internally; convert to Mpc/h for plots using h = H0/100.

**Power spectrum** (recommended for IC validation):
```bash
# 1. Generate CLASS theory P(k) at the IC redshift (edit z_pk in data/class_pk.ini first):
./music_build/_deps/class-build/class data/class_pk.ini   # → class_pk_z{z}_pk.dat

# 2. Measure P(k) from particles and overlay theory:
conda run -n cosmo python scripts/compute_pk.py data/ics_swift_n256_z127_L25.hdf5 \
    --theory data/class_pk_z127_pk.dat
```

`data/class_pk.ini` has `root = class_pk_z{z}_` so output goes to `class_pk_z{z}_pk.dat` in the working directory (move to `data/` after).
Always run CLASS at the **same redshift as the IC** (`z_pk = {ZSTART}` in MUSIC conf).

**Shot noise and k-range**: The measurable k-range is limited by when P(k,z) > P_shot = V/N × h³.
At high z, P(k) is suppressed by D(z)² ≈ 1/(1+z)², so P_shot wins at lower k for larger/coarser boxes:

| Config | P_shot [(Mpc/h)³] | Valid k range (z=45) |
|--------|-------------------|----------------------|
| 256³, L=500 Mpc/h | 7.45 | shot-noise dominated |
| 512³, L=500 Mpc/h | 0.93 | k ≲ 0.2 h/Mpc |
| 1024³, L=500 Mpc/h | 0.12 | k ≲ 0.7 h/Mpc |
| 256³, L=25 Mpc/h | 0.06 | k ≲ 3 h/Mpc |

**Correlation function** (useful at low z only; avoid at high z):
```bash
conda run -n cosmo python scripts/make_rbins.py --hdf5 data/ics_swift_n256_z127_L25.hdf5
./compute_xi data/ics_swift_n256_z127_L25.hdf5 data/rbins_n256_z127_L25.txt 8
```
At z ≳ 10 the cosmological signal in ξ(r) is ~10⁻⁶ — undetectable with subsampled pair counts.
Use P(k) instead. See CORRFUNC.md § "When ξ(r) is the wrong tool".

## prepare-music.sh

Uses `uname -s` to detect macOS vs cluster:
- **macOS (Darwin):** sets `FC=gfortran-14`, relies on Homebrew-installed fftw, gsl, hdf5, open-mpi
- **Cluster (non-Darwin):** loads modules via `module load gnu12 openmpi4 fftw hdf5 gsl cmake`

Skips compilation if `music_build/MUSIC` already exists.
