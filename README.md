# cosmo-pipeline

Scripts and configuration for cosmological IC generation with MUSIC2 and analysis.

## Environment

```bash
conda env create -f env.yml
conda activate cosmo
```

## IC Generation (MUSIC2)

```bash
# Build MUSIC2 (first time or after source changes)
./prepare-music.sh

# Run
./music_build/MUSIC CV_22_MUSIC.conf
```

Config: 25 Mpc/h box, 256³ particles, z=127, SWIFT output format.

## Scripts

| Script | Description |
|--------|-------------|
| `prepare-music.sh` | Builds MUSIC2 binary into `music_build/` |
| `read_ics_swift.py` | Reads and summarizes `ics_swift.hdf5` header and particle data |
| `read_wnoise.py` | Reads `wnoise_NNNN.bin` white noise binary into a numpy array |
| `plot_dr_histogram.py` | Plots histogram of particle displacement from lattice (dr/dx) |

## Configuration Files

| File | Description |
|------|-------------|
| `CV_22_MUSIC.conf` | MUSIC2 config: 25 Mpc/h, 256³, z=127 |
| `CV_22_MUSIC_z400.conf` | Same but z=400 (for comparison) |
