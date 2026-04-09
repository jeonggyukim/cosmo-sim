# cosmo-pipeline

Scripts and configuration for cosmological IC generation with MUSIC2 and IC validation.

## Directory layout

```
scripts/   — Python scripts
conf/      — MUSIC2 config files
data/      — CLASS outputs, rbins, measured P(k)/ξ tables  (HDF5/bin files gitignored)
plots/     — output figures  (gitignored)
notes/     — LaTeX write-ups and supporting plot scripts
```

## Setup

```bash
conda env create -f env.yml
conda activate cosmo
```

## Workflow

### 1. Build MUSIC2

```bash
./prepare-music.sh
```

Only needed once (or after source changes). Detects macOS vs cluster automatically.
Binary placed at `music_build/MUSIC`; CLASS built at `music_build/_deps/class-build/class`.

---

### 2. Generate a MUSIC2 config

```bash
conda run -n cosmo python scripts/make_music_conf.py -N 256 -z 45 -L 500
# → conf/CV_22_MUSIC_n256_z45_L500.conf
```

Arguments: `-N` particles per side, `-z` starting redshift, `-L` box size in Mpc/h.
Uses `conf/CV_22_MUSIC_template.conf` (CV_22 cosmology, SWIFT output format).

---

### 3. Run MUSIC2

```bash
./music_build/MUSIC conf/CV_22_MUSIC_n256_z45_L500.conf
# → data/ics_swift_n256_z45_L500.hdf5
```

MUSIC2 writes the IC file path embedded in the config (`data/ics_swift_n{N}_z{z}_L{L}.hdf5`).

---

### 4. Generate CLASS theory P(k)

Edit `data/class_pk.ini`: set `z_pk` to match the IC redshift, adjust `root` if needed.

```bash
./music_build/_deps/class-build/class data/class_pk.ini
# → class_pk_z{z}_pk.dat  (move to data/ after)
mv class_pk_z45_pk.dat data/
```

---

### 5. Validate ICs — power spectrum

```bash
conda run -n cosmo python scripts/compute_pk.py \
    data/ics_swift_n256_z45_L500.hdf5 \
    --theory data/class_pk_z45_pk.dat
# → plots/pk_n256_z45_L500.png
```

Produces two panels: P(k) with theory overlay, and ξ(r) via Hankel transform.
Markers: fundamental mode k_f, Nyquist k_Ny, Bragg peak 2k_Ny, shot noise P_shot,
BAO scale r_d from Eisenstein & Hu (1998).

Optional flags: `--ngrid`, `--nkbins`, `--H0`, `--Omega_m`, `--Omega_b`, `-o output.png`.

---

### 6. Validate ICs — correlation function (low z only)

At z ≳ 10 the signal is shot-noise dominated; use P(k) instead.

```bash
# Generate bin file
conda run -n cosmo python scripts/make_rbins.py \
    --hdf5 data/ics_swift_n256_z45_L500.hdf5
# → data/rbins_n256_z45_L500.txt

# Measure ξ(r) with Corrfunc (last arg = number of threads)
./compute_xi data/ics_swift_n256_z45_L500.hdf5 \
             data/rbins_n256_z45_L500.txt 8
```

---

### 7. Diagnostic plots

```bash
# Particle displacement histogram (dr/dx from lattice)
conda run -n cosmo python scripts/plot_dr_histogram.py \
    data/ics_swift_n256_z45_L500.hdf5
```

---

## Notes

`notes/ic_sampling_review.tex` reviews the IC sampling literature (Pen 1997, Sirko 2005,
Hahn & Abel 2011): P-sampled vs ξ-sampled methods, box window truncation errors,
and implications for MUSIC2/monofonIC.

```bash
cd notes && make    # compiles PDF and opens it
```
