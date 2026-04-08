# CLAUDE.md — cosmo-pipeline

This repo contains scripts and configuration files for running cosmological simulations, primarily IC generation with MUSIC2 and N-body/hydro runs with SWIFT.

## Key Files

- `prepare-music.sh` — builds MUSIC2 from source and places the binary in `music_build/`
- `CV_22_MUSIC.conf` — canonical MUSIC2 config: 25 Mpc/h box, 256^3 (level 8), z=127, SWIFT output
- `CV_22_MUSIC_template.conf` — template config with `{BOXLENGTH}`, `{ZSTART}`, `{LEVEL}`, `{FILENAME}` placeholders
- `make_music_conf.py` — generates a config from the template given N, z, L
- `make_rbins.py` — generates Corrfunc bin file appropriate for a given IC run (rmin=2×mean spacing, rmax=L/3)
- `compute_xi.c` / `compute_xi` — measures ξ(r) from IC particles using Corrfunc (C, compiled via `make`)
- `compute_pk.py` — estimates P(k) from IC particles via CIC + FFT, compares to CLASS theory
- `read_wnoise.py` — reads `wnoise_NNNN.bin` white noise binary into a numpy array
- `read_ics_swift.py` — reads `ics_swift.hdf5` IC file (header + particle data)
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
python make_music_conf.py -N 256 -z 127 -L 25

# Run
./music_build/MUSIC CV_22_MUSIC_n256_z127_L25.conf
```

## Python environment

Use `conda run -n cosmo python ...` for all Python scripts in this project.

## IC validation

SWIFT stores coordinates and BoxSize in **Mpc** (not Mpc/h). All bin files and P(k) calculations use Mpc internally; convert to Mpc/h for plots using h = H0/100.

**Power spectrum** (recommended for IC validation):
```bash
python compute_pk.py music_build/ics_swift_n256_z127_L25.hdf5
```
Overlay CLASS theory using `music_build/_deps/class-build/class` with `output = mPk`.

**Correlation function** (useful at low z; shot-noise dominated at high z for large boxes):
```bash
python make_rbins.py --hdf5 ics_swift_n256_z127_L25.hdf5   # generate bins first
./compute_xi ics_swift_n256_z127_L25.hdf5 rbins_n256_z127_L25.txt 8
```

## prepare-music.sh

Uses `uname -s` to detect macOS vs cluster:
- **macOS (Darwin):** sets `FC=gfortran-14`, relies on Homebrew-installed fftw, gsl, hdf5, open-mpi
- **Cluster (non-Darwin):** loads modules via `module load gnu12 openmpi4 fftw hdf5 gsl cmake`

Skips compilation if `music_build/MUSIC` already exists.
