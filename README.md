# cosmo-pipeline

Scripts and configuration for cosmological IC generation with MUSIC2 and IC validation.

## Directory layout

```
src/        — C source files (compute_xi.c, compute_xi_cic.c)
scripts/   — Python scripts
conf/      — MUSIC2 config files
data/      — CLASS outputs, rbins, measured P(k)/ξ/ψ tables  (HDF5/bin files gitignored)
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
./run_pipeline.sh                                          # defaults: N=256, L=687 Mpc/h (~1024 Mpc), z=200
./run_pipeline.sh --ngrid 256 --lbox 344 --zstart 200      # 256³, L=344 Mpc/h (~512 Mpc), z=200
./run_pipeline.sh --ngrid 256 --lbox 172 --zstart 200      # 256³, L=172 Mpc/h (~256 Mpc), z=200
```

The pipeline runs all steps below automatically, skipping any that are already complete.

Pipeline steps:
1. Build MUSIC2 (skipped if binary exists)
2. Build `compute_xi` (skipped if binary exists)
3. Generate MUSIC2 config from template
4. Run MUSIC2 → IC HDF5 + `input_class_parameters.ini`
5. Run CLASS → `data/class_pk_z{z}_pk.dat`
6. Generate Corrfunc radial bin file
7. Measure ξ(r) with Corrfunc pair counting
8. Measure ξ(r) and ψ(r) on CIC grid (`compute_xi_cic --vel`)
9. Measure P(k) with CIC+FFT (`compute_pk.py`)
10. Plot diagnostics (`plot_ic.py`) → `plots/pk_{STEM}.png`

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
conda run -n cosmo python scripts/make_music_conf.py -N 256 -z 2 -L 687
# → conf/CV_22_MUSIC_n256_z2_L687.conf
```

Arguments: `-N` particles per side, `-z` starting redshift, `-L` box size in Mpc/h.
Uses `conf/CV_22_MUSIC_template.conf` (CV_22 cosmology, SWIFT output format).

---

#### 3. Run MUSIC2

```bash
./music_build/MUSIC conf/CV_22_MUSIC_n256_z2_L687.conf
# → data/ics_swift_n256_z2_L687.hdf5
# → conf/input_class_parameters_n256_z2_L687.ini  (moved from repo root by pipeline)
```

SWIFT stores coordinates and BoxSize in **Mpc** (not Mpc/h). Velocities are stored as
`v_int = a × v_pec` (canonical momentum convention); divide by a² when computing ψ(r).

---

#### 4. Generate CLASS theory P(k)

The pipeline derives the CLASS ini from `input_class_parameters.ini` (written by MUSIC2),
changing `output` to `mPk` and setting `z_pk` to the IC redshift. No hand-edited ini needed.

```bash
# Manually (after MUSIC2 has run):
TMP=$(mktemp /tmp/class_pk_XXXXXX)
sed -e "s/^output =.*/output = mPk/" -e "s/^z_pk =.*/z_pk = 2/" \
    -e "/^extra metric transfer functions/d" -e "/^gauge/d" \
    conf/input_class_parameters_n256_z2_L687.ini > "$TMP"
echo "root = class_pk_z2_" >> "$TMP"
./music_build/_deps/class-build/class "$TMP"
mv class_pk_z2_pk.dat data/
```

---

#### 5. Validate ICs — power spectrum

```bash
conda run -n cosmo python scripts/compute_pk.py \
    data/ics_swift_n256_z2_L687.hdf5 \
    -o data/pk_n256_z2_L687.txt

conda run -n cosmo python scripts/plot_ic.py \
    data/pk_n256_z2_L687.txt \
    --theory data/class_pk_z2_pk.dat
# → plots/pk_n256_z2_L687.png
```

Overlay multiple box sizes:
```bash
conda run -n cosmo python scripts/plot_ic.py \
    data/pk_n256_z2_L172.txt data/pk_n256_z2_L344.txt data/pk_n256_z2_L687.txt \
    --theory data/class_pk_z2_pk.dat -o plots/comparison.png
```

---

#### 6. Validate ICs — correlation functions

```bash
# Corrfunc pair-counting ξ(r) (low z; shot-noise dominated at z ≳ 10):
conda run -n cosmo python scripts/make_rbins.py \
    --hdf5 data/ics_swift_n256_z2_L687.hdf5
# → data/rbins_n256_z2_L687.txt

./compute_xi data/ics_swift_n256_z2_L687.hdf5 \
             data/rbins_n256_z2_L687.txt 8 \
             > data/xi_n256_z2_L687.txt

# CIC grid ξ(r) and ψ(r) = ⟨v_pec·v_pec'⟩ (works at any z):
./compute_xi_cic \
    --input    data/ics_swift_n256_z2_L687.hdf5 \
    --Ngrid    128 \
    --nthreads 8 \
    --output   data/xi_cic_n256_z2_L687.txt \
    --vel
# → data/xi_cic_n256_z2_L687.txt
# → data/vel_cic_n256_z2_L687.txt
```

`plot_ic.py` auto-detects all `xi_*`, `xi_cic_*`, and `vel_cic_*` files alongside the pk file
and overlays them. Theory ψ(r) = [H(z)f(z)]²/(2π²) ∫ P(k) j₀(kr) dk is computed from
CLASS P(k); the measured ψ is corrected from SWIFT internal units (a·v_pec) to peculiar
velocity units automatically.

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
    data/ics_swift_n256_z2_L687.hdf5
```

---

## Notes

Three LaTeX write-ups live in `notes/`:

| File | Contents |
|------|----------|
| `cosmo_ic.tex` | Cosmological ICs: fluid equations, ZA, 2LPT derivation, Zel'dovich pancake, IC generation algorithm, starting redshifts, P(k)/ξ(r)/ψ(r), one-loop P(k); §11 implementation details for MUSIC2 and monofonIC (refinement hierarchy, hybrid Poisson solve, 2LPT/3LPT source, PLT, Orszag 3/2 rule, back-scaling). Appendix A: Fourier conventions. Appendix B: P(k) estimation from N-body (CIC window, deconvolution, shot noise). |
| `ic_sampling.tex` | IC sampling literature (Pen 1997, Sirko 2005, Hahn & Abel 2011): P-sampled vs ξ-sampled methods, box window truncation errors, implications for MUSIC2/monofonIC |
| `fft.tex` | FFT reference: DFT definition, Cooley–Tukey radix-2, FFTW mixed-radix, multi-dimensional row–column algorithm, MPI slab vs. pencil decomposition, FFTW API, fftMPI (Plimpton 2019) API and Tigris `BlockFFT` usage |

```bash
cd notes
make            # regenerate figures + compile all PDFs (opens automatically)
make figures    # regenerate figures only
make notes      # compile PDFs only (assumes figures exist)
make clean      # remove figures, PDFs, and LaTeX aux files
```

Figures are generated from `plot_box_window.py`, `plot_tophat_window.py`, and
`plot_pgrid.py` (the last requires `data/class_pk_z2_pk.dat`).
