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

### Run the full pipeline (recommended)

```bash
./run_pipeline.sh                                          # defaults: ngrid=512, lbox=687 (~1024 Mpc), zstart=45
./run_pipeline.sh --ngrid 512 --lbox 500 --zstart 127      # example: 512³, L=500 Mpc/h, z=127
```

The pipeline runs all steps below automatically, skipping any that are already complete.

---

### Step by step

#### 1. Build MUSIC2

```bash
./build-music.sh
```

Only needed once (or after source changes). Clones MUSIC2 from GitHub if the source directory
does not exist. Detects macOS vs cluster automatically.
Binary placed at `music_build/MUSIC`; CLASS built at `music_build/_deps/class-build/class`.

---

#### 2. Generate a MUSIC2 config

```bash
conda run -n cosmo python scripts/make_music_conf.py -N 256 -z 45 -L 500
# → conf/CV_22_MUSIC_n256_z45_L500.conf
```

Arguments: `-N` particles per side, `-z` starting redshift, `-L` box size in Mpc/h.
Uses `conf/CV_22_MUSIC_template.conf` (CV_22 cosmology, SWIFT output format).

---

#### 3. Run MUSIC2

```bash
./music_build/MUSIC conf/CV_22_MUSIC_n256_z45_L500.conf
# → data/ics_swift_n256_z45_L500.hdf5
# → input_class_parameters.ini  (written to repo root by MUSIC2's CLASS plugin)
```

---

#### 4. Generate CLASS theory P(k)

The pipeline derives the CLASS ini from `input_class_parameters.ini` (written by MUSIC2),
changing `output` to `mPk` and setting `z_pk` to the IC redshift. No hand-edited ini needed.

```bash
# Manually (after MUSIC2 has run):
TMP=$(mktemp /tmp/class_pk_XXXXXX)
sed -e "s/^output =.*/output = mPk/" -e "s/^z_pk =.*/z_pk = 45/" \
    -e "/^extra metric transfer functions/d" -e "/^gauge/d" \
    input_class_parameters.ini > "$TMP"
echo "root = class_pk_z45_" >> "$TMP"
./music_build/_deps/class-build/class "$TMP"
mv class_pk_z45_pk.dat data/
```

---

#### 5. Validate ICs — power spectrum

```bash
conda run -n cosmo python scripts/compute_pk.py \
    data/ics_swift_n256_z45_L500.hdf5 \
    --theory data/class_pk_z45_pk.dat
# → data/pk_n256_z45_L500.txt + plots/pk_n256_z45_L500.png
```

Produces two panels: P(k) with theory overlay, and ξ(r) via Hankel transform.
Markers: fundamental mode k_f, Nyquist k_Ny, Bragg peak 2k_Ny, shot noise P_shot,
BAO scale r_d from Eisenstein & Hu (1998).

To re-plot from saved `.txt` files (e.g. to overlay multiple runs):
```bash
conda run -n cosmo python scripts/plot_pk.py \
    data/pk_n256_z45_L500.txt data/pk_n512_z45_L500.txt \
    --theory data/class_pk_z45_pk.dat -o plots/comparison.png
```

---

#### 6. Validate ICs — correlation function (low z only)

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

#### 7. Clean generated outputs

```bash
./clean.sh          # remove plots, P(k)/xi tables, rbins, configs, CLASS outputs
./clean.sh --all    # also remove IC HDF5 files and wnoise binaries (slow to regenerate)
```

---

#### 8. Diagnostic plots

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
cd notes
make            # regenerate figures + compile PDF (opens automatically)
make figures    # regenerate figures only
make notes      # compile PDF only (assumes figures exist)
make clean      # remove figures, PDF, and LaTeX aux files
```

Figures are generated from `plot_box_window.py`, `plot_tophat_window.py`, and
`plot_pgrid.py` (the last requires `data/class_pk_z45_pk.dat`).
