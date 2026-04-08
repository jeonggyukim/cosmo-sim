# CLAUDE.md — cosmo-pipeline

This repo contains scripts and configuration files for running cosmological simulations, primarily IC generation with MUSIC2 and N-body/hydro runs with SWIFT.

## Key Files

- `prepare-music.sh` — builds MUSIC2 from source and places the binary in `music_build/`
- `CV_22_MUSIC.conf` — canonical MUSIC2 config: 25 Mpc/h box, 256^3 (level 8), z=127, SWIFT output
- `CV_22_MUSIC_template.conf` — template config with `{BOXLENGTH}`, `{ZSTART}`, `{LEVEL}`, `{FILENAME}` placeholders
- `make_music_conf.py` — generates a config from the template given N, z, L
- `read_wnoise.py` — reads `wnoise_NNNN.bin` white noise binary into a numpy array
- `read_ics_swift.py` — reads `ics_swift.hdf5` IC file (header + particle data)
- `music_build/MUSIC` — compiled MUSIC2 binary (gitignored)

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

## prepare-music.sh

Uses `uname -s` to detect macOS vs cluster:
- **macOS (Darwin):** sets `FC=gfortran-14`, relies on Homebrew-installed fftw, gsl, hdf5, open-mpi
- **Cluster (non-Darwin):** loads modules via `module load gnu12 openmpi4 fftw hdf5 gsl cmake`

Skips compilation if `music_build/MUSIC` already exists.
