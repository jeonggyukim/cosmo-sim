# cosmo-pipeline

End-to-end IC generation (MUSIC2) + IC validation (CIC P(k), Corrfunc
ξ(r), CIC ξ/ψ) for cosmological SWIFT simulations.

## Directory layout

| dir                 | purpose                                                  | sub-README |
|---------------------|----------------------------------------------------------|-----------|
| `src/`              | C sources + Makefile (`compute_xi`, `compute_xi_cic`)    | —         |
| `bin/`              | compiled C binaries (gitignored)                         | —         |
| `icpipe/`           | Python library: `ICField`, `LinearTheory`, `io`          | [`icpipe/README.md`](icpipe/README.md) |
| `icpipe/cli/`       | pipeline-step console scripts                            | (in `icpipe/README.md`) |
| `scripts/analysis/` | ad-hoc inspection CLIs (not in `run_pipeline.sh`)        | [`scripts/analysis/README.md`](scripts/analysis/README.md) |
| `tools/`            | build / clean / cluster sbatch scripts                   | [`tools/README.md`](tools/README.md) |
| `conf/`             | MUSIC2 config files (template + generated)               | —         |
| `data/`             | CLASS outputs, rbins, measured P/ξ/ψ tables (gitignored) | —         |
| `plots/`            | output figures (gitignored)                              | —         |
| `notes/`            | LaTeX write-ups                                          | (sources) |
| `notes/figures/`    | plot scripts feeding the LaTeX figures                   | [`notes/figures/README.md`](notes/figures/README.md) |
| `notebooks/`        | worked examples using `icpipe`                           | [`notebooks/README.md`](notebooks/README.md) |
| `tests/validation/` | bundled validation tests (e.g. matched-noise zoom)       | (per-test) |

## Setup

```bash
conda env create -f env.yml
conda activate cosmo
pip install -e ".[plot,test]"     # installs icpipe + the pipeline-step CLIs onto $PATH
```

The `pip install -e .` step also wires `make-music-conf`, `make-rbins`,
`compute-pk`, `compute-pv`, and `plot-ic` as commands on `$PATH`.  See
[`icpipe/README.md`](icpipe/README.md) for the full library API and CLI table.

## Run the full pipeline

```bash
./run_pipeline.sh                                          # defaults: N=256, L=1000 Mpc/h, z=200
./run_pipeline.sh --ngrid 256 --lbox 512 --zstart 200      # 256³, L=512 Mpc/h, z=200
./run_pipeline.sh --ngrid 256 --lbox 256 --zstart 200      # 256³, L=256 Mpc/h, z=200
```

Each step is skipped if its output already exists:

1. Build MUSIC2 (clones source if absent — `tools/build-music.sh`)
2. Build `compute_xi` / `compute_xi_cic` C binaries → `bin/`
3. Generate MUSIC2 config from `conf/CV_22_MUSIC_template.conf`
4. Run MUSIC2 → `data/ics_swift_*.hdf5` + `conf/input_class_parameters_*.ini`
5. Run CLASS → `data/class_pk_z{z}_pk.dat`
6. Generate Corrfunc radial bin file
7. Measure ξ(r) with Corrfunc pair counting (skipped at z ≳ 10; shot-noise dominated)
8. Measure ξ(r) and ψ(r) on a CIC grid (`bin/compute_xi_cic --vel`)
9. Measure P(k) with CIC + FFT (`compute-pk`)
10. Plot diagnostics (`plot-ic`) → `plots/pk_{stem}.png`

Clean up with `tools/clean.sh` (or `tools/clean.sh --all` to also wipe ICs / wnoise / MUSIC2 build).

## Run a single step

Each pipeline-step CLI is also runnable on its own.  See
[`icpipe/README.md`](icpipe/README.md) for command-line flags;
typical invocations:

```bash
make-music-conf -N 256 -z 200 -L 1024                          # writes conf/CV_22_MUSIC_n256_z200_L1024.conf
./music_build/MUSIC conf/CV_22_MUSIC_n256_z200_L1024.conf      # runs MUSIC2
compute-pk data/ics_swift_n256_z200_L1024.hdf5 \
    -o data/pk_n256_z200_L1024.txt                             # CIC + FFT
make-rbins --hdf5 data/ics_swift_n256_z200_L1024.hdf5          # Corrfunc bin edges
./bin/compute_xi_cic --input data/ics_swift_n256_z200_L1024.hdf5 \
    --nthreads 8 --output data/xi_cic_n256_z200_L1024.txt --vel
plot-ic data/pk_n256_z200_L1024.txt \
    --theory data/class_pk_z0_pk.dat --theory-zref 0           # plots/pk_*.png
```

`plot-ic` auto-detects `xi_*.txt`, `xi_cic_*.txt`, and `vel_cic_*.txt`
alongside the input `pk_*.txt` and overlays them.  Pass several
`pk_*.txt` files to overlay multiple box sizes.

## Units & conventions (gotchas)

- SWIFT stores coordinates and `BoxSize` in **Mpc** (not Mpc/h).  All
  internal calculations use Mpc; conversion to Mpc/h uses `h = H0/100`.
- The on-disk `Velocities` dataset is the **peculiar velocity in km/s**
  (`v_pec = a·H·f·Ψ`).  SWIFT itself converts to its internal
  canonical-momentum variable `u = a·v_pec` only at IC read time.  So
  ψ(r) measured by `bin/compute_xi_cic` is already in (km/s)² — no
  a² rescaling.
- MUSIC2 always writes `wnoise_NNNN.bin` and `input_class_parameters.ini`
  to the CWD; `run_pipeline.sh` moves them into `data/` and `conf/`
  automatically.
- Always generate rbins per-IC with `make-rbins` (rmin = 2 × mean
  spacing, rmax = L/3).  Reusing rbins from a different box size or
  resolution silently produces empty bins or periodic-boundary artifacts.

## Notes

LaTeX write-ups under `notes/`:

| note                     | contents |
|--------------------------|----------|
| `fft.tex`                | FFT reference: DFT, Cooley–Tukey, FFTW, multi-dim row–column, MPI slab vs. pencil, `BlockFFT` |
| `cosmo_ic.tex`           | Cosmological ICs: fluid eqs, ZA, 2LPT, pancake, IC generation, P(k)/ξ(r)/ψ(r), MUSIC2 / monofonIC internals |
| `ic_sampling.tex`        | Pen 1997 / Sirko 2005 / Hahn 2011 IC sampling methods; box window truncation |
| `restriction_lpt.tex`    | Restriction of δ, Ψ⁽¹⁾, deformation, S⁽²⁾, φ⁽¹⁾; zoom-IC application |
| `music2_internals.tex`   | MUSIC2 internals walk-through; GRF appendix; Meyer-window figure |

Build all with:

```bash
make -C notes              # regenerate figures + compile all PDFs (opens automatically)
make -C notes figures      # figures only
make -C notes notes        # compile PDFs only
make -C notes clean        # remove figures, PDFs, LaTeX aux
```

See [`notes/figures/README.md`](notes/figures/README.md) for the
figure-script layout.
