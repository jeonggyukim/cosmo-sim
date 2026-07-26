# cosmo-sim

Generate cosmological initial conditions with **MUSIC2** and validate them.
The pipeline measures the density power spectrum **P(k)** (CIC + FFT), the
two-point correlation **ξ(r)** (CIC-grid FFT autocorrelation), and the velocity
correlation **ψ(r)**, then compares them against linear theory from **CLASS**.

The reusable analysis code is the installable Python package **`icpipe`**
(`ICField`, `LinearTheory`, `io`, and the pipeline-step CLIs). The end-to-end
run is driven by `run-pipeline`; the C tool `compute_xi` measures ξ(r) and
ψ(r) on a CIC grid at any z, and needs only HDF5 and FFTW.

## Install

`icpipe` is a standard `pyproject.toml` package. Pick one of the three options
below — all use an editable install (`-e`) so edits to `icpipe/` take effect
without reinstalling. Installing also wires the pipeline-step CLIs onto `$PATH`:
`make-music-conf`, `compute-pk`, `compute-pv`, `plot-ic`.

**Option 1 — conda + pip** (recommended; full analysis environment)
```bash
conda env create -f env.yml        # creates the `cosmo` env with all dependencies
conda activate cosmo
pip install -e . --no-deps          # installs icpipe + its CLIs; conda provides the deps
```

**Option 2 — conda + [uv](https://docs.astral.sh/uv/)** (same env, faster editable resolve)
```bash
conda env create -f env.yml
conda activate cosmo
uv pip install -e . --no-deps       # uv uses the active conda env
```

**Option 3 — uv only (no conda)**
```bash
uv venv --python 3.12
source .venv/bin/activate           # Windows: .venv\Scripts\activate
uv pip install -e ".[plot]"         # numpy/scipy/h5py + matplotlib/mcfit from PyPI
```

`--no-deps` on the conda paths keeps pip/uv from pulling PyPI wheels over the
conda-forge binaries (`h5py`, `numpy`, …); there `env.yml` is the dependency
source. The uv-only path resolves every dependency from PyPI via `pyproject.toml`
(add the `test` extra — `".[plot,test]"` — to run the test suite).

Sanity check any option:
```bash
python -c "import icpipe; print(icpipe.__file__)"
pytest icpipe/tests/                # 17 tests (needs the `test` extra)
```

Note: the uv-only environment has the `icpipe` library but not the conda-only
analysis extras (`astropy`, `jupyter`) or the compiled C tools. The
`bin/compute_xi` binary and the MUSIC2 / CLASS builds are separate and need
system libraries — see `tools/build-music.sh` and
[`tools/README.md`](tools/README.md).

## Uninstall

```bash
pip uninstall icpipe                 # or: uv pip uninstall icpipe
```
Remove the whole conda environment with `conda env remove -n cosmo`, or delete
the uv virtualenv with `rm -rf .venv`. Generated pipeline outputs are cleared
separately with `tools/clean.sh` (`--all` also removes ICs, wnoise binaries,
and the MUSIC2 build).

## Directory layout

| dir                 | purpose                                                  | sub-README |
|---------------------|----------------------------------------------------------|-----------|
| `src/`              | C source + Makefile (`compute_xi`)                       | [`src/README.md`](src/README.md) |
| `bin/`              | compiled C binaries (gitignored)                         | —         |
| `icpipe/`           | Python library: `ICField`, `LinearTheory`, `io`          | [`icpipe/README.md`](icpipe/README.md) |
| `icpipe/cli/`       | pipeline-step console scripts                            | (in `icpipe/README.md`) |
| `scripts/analysis/` | ad-hoc inspection CLIs (not in `run-pipeline`)        | [`scripts/analysis/README.md`](scripts/analysis/README.md) |
| `tools/`            | build / clean / cluster sbatch scripts                   | [`tools/README.md`](tools/README.md) |
| `conf/`             | MUSIC2 config files (template + generated)               | —         |
| `data/`             | CLASS outputs, measured P/ξ/ψ tables (gitignored)        | —         |
| `plots/`            | output figures (gitignored)                              | —         |
| `notes/`            | LaTeX write-ups                                          | (sources) |
| `notes/figures/`    | plot scripts feeding the LaTeX figures                   | [`notes/figures/README.md`](notes/figures/README.md) |
| `notebooks/`        | worked examples using `icpipe`                           | [`notebooks/README.md`](notebooks/README.md) |
| `tests/validation/` | bundled validation tests (e.g. matched-noise zoom)       | (per-test) |

## Run the full pipeline

Two IC codes are supported, selected with `--ic-code` on the driver
`run-pipeline`.

```bash
run-pipeline                                     # MUSIC (default): N=256, L=1000 Mpc/h, z=200
run-pipeline --ic-code monofonic                 # MUSIC2-monofonIC (adds PLT, 3LPT)
run-pipeline --ngrid 256 --lbox 512 --zstart 200 # 256³, L=512 Mpc/h, z=200
run-pipeline --ic-code monofonic --lpt-order 3   # explicit LPT order
```

`music` and `monofonic` runs at the same N, z, L coexist — monofonIC outputs
carry a `_mono` stem tag. **monofonIC** (Michaux et al. 2020) is the unigrid
successor to MUSIC: it adds the PLT (particle linear theory) correction, on via
`tools/build-monofonic.sh` (`-DENABLE_PLT=ON`), which fixes the near-Nyquist mode
amplitudes that legacy ZA/2LPT gets wrong on a particle lattice. MUSIC remains the
tool for nested zoom ICs. The downstream validation is identical for both codes.

Each step is skipped if its output already exists:

1. Build MUSIC2 (clones source if absent — `tools/build-music.sh`)
2. Build the `compute_xi` C binary (CIC-grid ξ/ψ; needs only HDF5 + FFTW) → `bin/`
3. Generate MUSIC2 config from `conf/CV_22_MUSIC_template.conf`
4. Run MUSIC2 → `data/ics_swift_*.hdf5` + `conf/input_class_parameters_*.ini`
5. Run CLASS → `data/class_pk_z{z}_pk.dat`
6. Measure ξ(r) and ψ(r) on a CIC grid (`bin/compute_xi --vel`; works at any z)
7. Measure P(k) with CIC + FFT (`compute-pk`)
8. Plot diagnostics (`plot-ic`) → `plots/pk_{stem}.png`

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
./bin/compute_xi --input data/ics_swift_n256_z200_L1024.hdf5 \
    --nthreads 8 --output data/xi_cic_n256_z200_L1024.txt --vel
plot-ic data/pk_n256_z200_L1024.txt \
    --theory data/class_pk_z0_pk.dat --theory-zref 0           # plots/pk_*.png
```

`plot-ic` auto-detects `xi_cic_*.txt` and `vel_cic_*.txt` alongside the
input `pk_*.txt` and overlays them.  Pass several
`pk_*.txt` files to overlay multiple box sizes.

## Units & conventions (gotchas)

- SWIFT stores coordinates and `BoxSize` in **Mpc** (not Mpc/h).  All
  internal calculations use Mpc; conversion to Mpc/h uses `h = H0/100`.
- The on-disk `Velocities` dataset is the **peculiar velocity in km/s**
  (`v_pec = a·H·f·Ψ`).  SWIFT itself converts to its internal
  canonical-momentum variable `u = a·v_pec` only at IC read time.  So
  ψ(r) measured by `bin/compute_xi` is already in (km/s)² — no
  a² rescaling.
- MUSIC2 always writes `wnoise_NNNN.bin` and `input_class_parameters.ini`
  to the CWD; `run-pipeline` moves them into `data/` and `conf/`
  automatically.

## Notes

LaTeX write-ups under `notes/`:

| note                     | contents |
|--------------------------|----------|
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

## License

Two licenses, because this repository holds two kinds of material:

| Material | License |
|----------|---------|
| Code — `icpipe/`, `src/`, `tools/`, `scripts/`, `notebooks/`, `conf/`, and the plotting scripts under `notes/figures/` | [MIT](LICENSE) |
| Notes — `notes/*.tex` and the PDFs built from them | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

`notes/aasjournal.bst` is an American Astronomical Society BibTeX style,
included unmodified because it is needed to build the notes and is not in
TeX Live under that name. It is not covered by either license above.

The external codes this pipeline drives — MUSIC2, MUSIC2-monofonIC, CLASS and
SWIFT — are not included here and carry their own licenses.

Every note carries a disclaimer on its first page: these documents were
compiled with AI assistance and have not been reviewed by a human domain
expert. Check derivations against the primary literature before relying on
them.
