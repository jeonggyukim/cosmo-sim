# CLAUDE.md — cosmo-pipeline

This repo contains scripts and configuration files for running cosmological simulations, primarily IC generation with MUSIC2 and N-body/hydro runs with SWIFT.

## Directory Structure

```
cosmo-pipeline/
  src/                — C source files + Makefile (compute_xi.c, compute_xi_cic.c)
  bin/                — compiled C binaries (gitignored)
  icpipe/             — installable Python library (ICField, LinearTheory, io)
  icpipe/cli/         — pipeline-step CLI modules exposed as console_scripts
                        (make-music-conf, make-monofonic-conf, make-rbins,
                         compute-pk, compute-pv, plot-ic)
  scripts/analysis/   — ad-hoc inspection / one-off CLIs
                        (check_ic, plot_box_size_comparison, plot_dr_histogram)
  tools/              — build/clean shell scripts + cluster sbatch templates
                        (build-music.sh, build-monofonic.sh, build-corrfunc.sh,
                         clean.sh, mpirun_restart.sbatch)
  data/               — IC HDF5 files, CLASS P(k) outputs, rbins files, wnoise binaries, measured P/xi tables
  plots/       — PNG/PDF figures (pk_*.png, xi_*.png, ...)
  conf/        — MUSIC2 configs (CV_22_MUSIC*.conf) and log files
  notes/       — LaTeX write-ups and supporting plot scripts
  music_build/ — compiled MUSIC2 binary (gitignored)
```

Note: `data/*.hdf5` and `data/*.bin` are gitignored (large binary files).

## Python Environment

Use `conda run -n cosmo python ...` for all Python scripts in this project.

## Pipeline: Quick Start

Two IC codes are supported. Use the Python driver `run_pipeline.py` with
`--ic-code`; the legacy bash `run_pipeline.sh` (MUSIC only) is kept as a fallback.

```bash
conda run -n cosmo python run_pipeline.py                                   # MUSIC (default), N=256, L=1000, z=200
conda run -n cosmo python run_pipeline.py --ic-code monofonic              # MUSIC2-monofonIC (PLT, 3LPT)
conda run -n cosmo python run_pipeline.py --ngrid 256 --lbox 512 --zstart 200
conda run -n cosmo python run_pipeline.py --ic-code monofonic --lpt-order 3

# Point the IC-code build at an existing source checkout (else sibling dir or clone)
conda run -n cosmo python run_pipeline.py --ic-source-dir /path/to/MUSIC2 --corrfunc-dir /path/to/Corrfunc

# Legacy bash driver (MUSIC only)
./run_pipeline.sh --ngrid 256 --lbox 512 --zstart 200
```

`--ic-code music` and `--ic-code monofonic` runs at the same N, z, L coexist:
monofonIC outputs carry a `_mono` tag in every stem (e.g.
`data/ics_swift_n256_z200_L1000_mono.hdf5`). Both codes write a CLASS ini during
the run (MUSIC: `input_class_parameters.ini`; monofonIC:
`<config-basename>_input_class_parameters.ini`, prefixed and written to the CWD);
the driver normalizes it to `conf/input_class_parameters_{stem}.ini` and adapts
it for the matter-P(k) CLASS run. The result `data/class_pk_z0_pk.dat` is
cosmology-keyed and reused across runs (identical CV_22 cosmology). monofonIC
links CLASS as a library and builds no standalone `class` binary, so a
monofonic-only setup regenerates the theory with MUSIC's CLASS binary or a
prebuilt `class_pk` (else the theory overlay is skipped).

**Thread count**: `run_pipeline.sh` defaults `--nthreads` to all logical CPUs on the current machine (detected via `sysctl -n hw.logicalcpu` on macOS or `nproc` on Linux). Override with `--nthreads N` to match a cluster job allocation.

Pipeline steps (each skipped if output already exists):
1. Build MUSIC2 binary
2. Build `compute_xi` and `compute_xi_cic`
3. Generate MUSIC2 config from template
4. Run MUSIC2 → IC HDF5 + `input_class_parameters.ini`
5. Run CLASS → `data/class_pk_z{z}_pk.dat`
6. Generate Corrfunc radial bin file
7. Measure ξ(r) with Corrfunc pair counting
8. Measure ξ(r) and ψ(r) on CIC grid (`compute_xi_cic --vel`)
9. Measure P(k) with CIC+FFT (`compute_pk.py`)
10. Plot diagnostics (`plot_ic.py`) → `plots/pk_{STEM}.png`

Remove outputs with `tools/clean.sh` (or `--all` to also remove IC HDF5 files).

## Key Files

### Pipeline scripts
- `run_pipeline.py` — unified Python driver; `--ic-code {music,monofonic}` selects
  the IC generator. Front-end (build → config → run) branches on the code; the
  downstream (CLASS P(k) → ξ(r) → CIC ξ/ψ → P(k) → plot) is code-agnostic (both
  write SWIFT HDF5). Preferred entry point.
- `run_pipeline.sh` — legacy bash pipeline (MUSIC only); kept as a fallback
- `tools/clean.sh` — remove generated outputs; `--all` also removes IC HDF5 files and wnoise binaries
- `tools/build-music.sh` — builds legacy MUSIC from source (see below)
- `tools/build-monofonic.sh` — builds MUSIC2-monofonIC from source into
  `monofonic_build/` (binary `monofonic_build/monofonIC`) with `-DENABLE_PLT=ON`
  (monofonIC defaults PLT OFF), `-DENABLE_MPI=ON` (required — monofonIC's source
  does not compile without MPI; the binary still runs single-rank, no mpirun),
  PANPHASIA off, CLASS on. macOS deps: Homebrew gcc, gsl, open-mpi, FFTW3-with-MPI,
  parallel `hdf5-mpi` (or the conda env's serial HDF5). Source resolution:
  `$MONOFONIC_SOURCE_DIR`, `../monofonIC`, else clone. CLASS is fetched
  automatically by monofonIC's CMake (FetchContent).

### IC generation (MUSIC2 / CLASS)
- `conf/CV_22_MUSIC_template.conf` — MUSIC template config with `{BOXLENGTH}`, `{ZSTART}`, `{LEVEL}`, `{FILENAME}` placeholders
- `conf/CV_22_MUSIC.conf` — canonical MUSIC2 config: 25 Mpc/h box, 256³ (level 8), z=127, SWIFT output
- `icpipe/cli/make_music_conf.py` (`make-music-conf`) — generates a MUSIC config from the template given N, z, L
- `conf/CV_22_monofonIC_template.conf` — monofonIC template with `{GRIDRES}`, `{BOXLENGTH}`,
  `{ZSTART}`, `{LPTORDER}`, `{DOFIXING}`, `{SEED}`, `{NTHREADS}`, `{FILENAME}` placeholders;
  same CV_22 cosmology as the MUSIC template
- `icpipe/cli/make_monofonic_conf.py` (`make-monofonic-conf`) — generates a monofonIC config;
  flags `--lpt-order` (1/2/3, default 3), `--fixing` (DoFixing), `--seed`, `--nthreads`.
  Output IC stem carries a `_mono` tag
- `input_class_parameters.ini` — written by MUSIC2 to repo root during each run; adapted by the pipeline for CLASS P(k) (gitignored)
- `data/class_pk_z{z}_pk.dat` — CLASS P(k) output at redshift z; generated by `run_pipeline.sh` (gitignored)
- `music_build/MUSIC` — compiled MUSIC2 binary (gitignored)
- `music_build/_deps/class-build/class` — CLASS binary (built by MUSIC2's CMake)

### Python package: `icpipe`
Reusable analysis library for IC fields. Install once with `pip install -e .` from the project root.
- `icpipe.ICField(hdf5, ngrid=..., interlace=True, h=..., load_velocities=True)` — main class. Loads positions/velocities, exposes cached CIC density/momentum/velocity Fourier fields, and `.power(field, ...)` for δ or velocity spectra.
- `icpipe.LinearTheory.from_class(class_pk_dat, z=..., h=..., Omega_m=...)` — linear-theory `Pk`, `Pv`, `xi`, `psi` from a single CLASS table (units consistent across all four).
- `icpipe.io.read_pk` / `read_pv` / `write_pk` / `write_pv` — ASCII I/O for the table formats below.
- Tests: `pytest icpipe/tests/` (17 tests).
- Examples: `notebooks/01_quickstart.ipynb` and `notebooks/02_box_size_sensitivity.ipynb`.

### IC validation
- `icpipe/cli/compute_pk.py` (`compute-pk`) — estimates P(k) from IC particles via CIC + FFT; saves `pk_*.txt` (no plotting). Thin wrapper around `icpipe.ICField.power('delta')`.
- `icpipe/cli/compute_pv.py` (`compute-pv`) — estimates the velocity power spectrum P_v(k) via CIC velocity assignment + FFT; saves `pv_*.txt` (k[h/Mpc], Pv_raw[(km/s)²(Mpc/h)³], Pv_nodeconv, sigma_Pv, nmodes). Uses `icpipe.ICField.power('velocity')`.
- `scripts/analysis/plot_box_size_comparison.py` — 4-panel figure (P_δ, P_v, ξ, ψ across box sizes) showing density-vs-velocity sensitivity to L.
- `icpipe/cli/plot_ic.py` (`plot-ic`) — IC diagnostics plotter: reads `pk_*.txt`, auto-detects `xi_*.txt` / `xi_cic_*.txt` / `vel_cic_*.txt`; overlays CLASS theory P(k), ξ(r), ψ(r); class-based (`ICPlotter`)
- `icpipe/cli/make_rbins.py` (`make-rbins`) — generates Corrfunc bin file appropriate for a given IC run (rmin=2×mean spacing, rmax=L/3)
- `src/compute_xi.c` → `bin/compute_xi` — measures ξ(r) from IC particles using Corrfunc (C, compiled via `make -C src`)
- `src/compute_xi_cic.c` → `bin/compute_xi_cic` — measures ξ(r) and ψ(r) via CIC density/velocity grid autocorrelation (C, compiled via `make -C src`); use `--vel` for velocity correlation

### Utilities
- `icpipe.io.read_wnoise(path)` — reads `wnoise_NNNN.bin` white noise binary into a numpy array
- `icpipe.io.read_swift_ics(path)` — returns `(header, parts)` from `ics_swift.hdf5` (all PartType groups + Header attrs); use `icpipe.io.print_swift_ics_summary(path)` for the pretty-printer
- `scripts/analysis/check_ic.py` — verifies DC (k=0) modes of displacement/velocity (Lagrangian + Eulerian CIC); prints mass resolution and recommended SWIFT force-softening range (ε = Δx/40…Δx/25). Add `--explain` for the preamble; `--hist PNG` to plot v_x/v_y/v_z (km/s) and δ histograms.

## IC Validation

SWIFT stores coordinates and BoxSize in **Mpc** (not Mpc/h). All bin files and P(k) calculations use Mpc internally; convert to Mpc/h for plots using h = H0/100.

### Power spectrum (recommended)

```bash
# Full pipeline (handles CLASS P(k) automatically):
./run_pipeline.sh --ngrid 256 --lbox 1024 --zstart 200

# Manually — two steps:
conda run -n cosmo compute-pk data/ics_swift_n256_z200_L1024.hdf5
conda run -n cosmo plot-ic data/pk_n256_z200_L1024.txt \
    --theory data/class_pk_z0_pk.dat --theory-zref 0

# Overlay multiple box sizes:
conda run -n cosmo plot-ic \
    data/pk_n256_z200_L256.txt data/pk_n256_z200_L512.txt data/pk_n256_z200_L1024.txt \
    --theory data/class_pk_z0_pk.dat --theory-zref 0
```

CLASS P(k) is generated by the pipeline from `input_class_parameters.ini` (written by MUSIC2 to repo root), with `output` changed to `mPk` and `z_pk` set to match the IC redshift. The output goes to `data/class_pk_z{z}_pk.dat`.

`plot_ic.py` auto-detects `xi_{stem}.txt`, `xi_cic_{stem}.txt`, and `vel_cic_{stem}.txt` in the same directory as the pk file and overlays them. It also computes theory ψ(r) from CLASS P(k).

**SWIFT velocity convention**: MUSIC2's SWIFT plugin writes the on-disk `Velocities` dataset in
peculiar-velocity units (km/s) — `v_out = a · H(a) · f · Ψ_comoving = v_pec`. SWIFT itself converts
to its internal canonical-momentum variable `u = a · v_pec` at IC read time (via the `Cosmology:`
flags in the SWIFT params), but the HDF5 array is v_pec. So the raw ψ(r) from `compute_xi_cic`
is already in (km/s)² and can be compared directly to the linear theory prediction
ψ_theory(r) = [H(z)f(z)]²/(2π²) ∫ P(k) j₀(kr) dk — no a² rescaling needed.

**Shot noise and k-range**: The measurable k-range is limited by when P(k,z) > P_shot = V/N × h³.
At high z, P(k) is suppressed by D(z)² ≈ 1/(1+z)², so P_shot wins at lower k for larger/coarser boxes:

| Config | P_shot [(Mpc/h)³] | Valid k range (z=45) |
|--------|-------------------|----------------------|
| 256³, L=500 Mpc/h | 7.45 | shot-noise dominated |
| 512³, L=500 Mpc/h | 0.93 | k ≲ 0.2 h/Mpc |
| 1024³, L=500 Mpc/h | 0.12 | k ≲ 0.7 h/Mpc |
| 256³, L=25 Mpc/h | 0.06 | k ≲ 3 h/Mpc |

### Correlation function (low z only; avoid at z ≳ 10)

```bash
# Pair counting (Corrfunc):
conda run -n cosmo make-rbins --hdf5 data/ics_swift_n256_z2_L172.hdf5
./bin/compute_xi data/ics_swift_n256_z2_L172.hdf5 data/rbins_n256_z2_L172.txt 8 \
    > data/xi_n256_z2_L172.txt

# CIC grid ξ(r) and ψ(r) (works at any z):
./bin/compute_xi_cic --input data/ics_swift_n256_z200_L1024.hdf5 \
    --nthreads 8 --output data/xi_cic_n256_z200_L1024.txt --vel
```

At z ≳ 10 the cosmological signal in pair-counting ξ(r) is ~10⁻⁶ — undetectable.
The CIC grid estimator can measure ξ(r) and ψ(r) at any z because it exploits
sub-Poissonian lattice shot noise (σ_δ ≈ D(z)·σ₀ ≪ 1). See `docs-claude/CLAUDE_CORRFUNC.md` § "When ξ(r) is the wrong tool".

**Rbins**: SWIFT stores coordinates in **Mpc** (not Mpc/h); rbins must be in Mpc to match.
Always generate rbins for each IC run with `make_rbins.py` — do not reuse a file from a different
box size or resolution. The script sets rmin = 2 × mean particle spacing (d = BoxSize/N) and
rmax = L/3; using smaller rmin gives empty bins (ξ = −1) and using larger rmax suffers periodic
boundary artifacts.

## Notes / Documentation

Three LaTeX write-ups live in `notes/`:

| File | Contents |
|------|----------|
| `cosmo_ic.tex` | Cosmological ICs: fluid equations, ZA, 2LPT, Zel'dovich pancake, IC generation, starting redshifts, P(k)/ξ(r)/ψ(r), one-loop P(k); §11 MUSIC2 / monofonIC implementation (refinement hierarchy, hybrid Poisson solve, 2LPT/3LPT source, PLT, Orszag 3/2 rule, back-scaling). Appendix A: Fourier conventions (CFT, Fourier series, DFT, Hermitian symmetry); see `fft.pdf` for implementation details. Appendix B: P(k) estimation (CIC window, deconvolution, shot noise, sub-Poissonian lattice ICs) |
| `ic_sampling.tex` | IC sampling methods (Pen 1997, Sirko 2005, Hahn & Abel 2011): P-sampled vs ξ-sampled, box window truncation errors |
| `fft.tex` | FFT reference: DFT, Cooley–Tukey radix-2, FFTW mixed-radix, multi-D row-column algorithm, MPI slab vs. pencil decomposition, FFTW API, fftMPI (Plimpton 2019) and Tigris `BlockFFT` usage |

Planning docs (markdown, kept next to the notes):

- `docs-claude/baryon_ic_plan.md` — roadmap for extending the pipeline from CDM-only to baryon+CDM ICs (theory write-up for `cosmo_ic.tex` §9, `--baryons` switch in `make_music_conf.py`, per-species P(k) diagnostics, two-species coherence test).

```bash
cd notes
make -C notes              # regenerate figures + compile all PDFs (opens automatically)
make -C notes figures      # regenerate figures only
make -C notes notes        # compile PDFs only (assumes figures exist)
make -C notes clean        # remove figures, PDFs, and LaTeX aux files
```

Figures are generated from `plot_box_window.py`, `plot_tophat_window.py`, and `plot_pgrid.py`
(the last requires `data/class_pk_z2_pk.dat`).

## MUSIC2

See [`docs-claude/CLAUDE_MUSIC2.md`](docs-claude/CLAUDE_MUSIC2.md) for full details on the MUSIC2 code structure, build
instructions, IC generation pipeline, transfer function internals, and file formats.

### Quick usage

```bash
# Build (first time or after source changes); MUSIC2 source auto-resolved
tools/build-music.sh

# Use a specific MUSIC2 source directory
MUSIC2_SOURCE_DIR=/path/to/MUSIC2 tools/build-music.sh

# Generate a config (canonical CV_22 run: N=256, z=127, L=25 Mpc/h)
conda run -n cosmo make-music-conf -N 256 -z 127 -L 25

# Run (config is written to conf/ by make_music_conf.py)
./music_build/MUSIC conf/CV_22_MUSIC_n256_z127_L25.conf
# Note: MUSIC2 always writes wnoise_NNNN.bin and input_class_parameters.ini
# to the CWD (paths are hardcoded in source).  run_pipeline.sh moves both
# to data/ and conf/ automatically.  For direct MUSIC2 invocations, move
# them manually: mv wnoise_*.bin data/  &&  mv input_class_parameters.ini conf/
```

### tools/build-music.sh

MUSIC2 source directory resolution order:
1. `$MUSIC2_SOURCE_DIR` environment variable (if set)
2. `../MUSIC2` — sibling of the repo root (default)
3. Clones from `https://github.com/cosmo-sims/MUSIC2` into `../MUSIC2` if not found

Uses `uname -s` to detect macOS vs cluster:
- **macOS (Darwin):** sets `FC=gfortran-14`, relies on Homebrew-installed fftw, gsl, hdf5, open-mpi
- **Cluster (non-Darwin):** loads modules via `module load gnu12 openmpi4 fftw hdf5 gsl cmake`

Skips compilation if `music_build/MUSIC` already exists.

### Matched-noise validation (MUSIC2-anisotropic-zoom fork)

For Hahn 2011 §4.3 matched-noise zoom-vs-unigrid validation in the
`MUSIC2-anisotropic-zoom` fork, the working combination is:

- `[random]/kaveraging = no` — makes MUSIC's level-N white noise coord-deterministic (per-cell RNG keyed on `(seed[N], i, j, k)`), no Meyer-window FFT splice across array shapes.
- `[setup]/density_boundary = yes` — Hahn 2011 §2.3.3 three-term density assembly at the coarse-fine boundary.

With both options set and the same `seed[levelmax]` integer in the zoom and the
single-level (`levelmin=levelmax`) unigrid conf, the patch-interior δ(q) rms
residual is 2.6×10⁻⁴σ at margin = 24 (commit `b90cfad` in
`MUSIC2-anisotropic-zoom`).  Do NOT try to pass the zoom's `wnoise_NNNN.bin` as
`seed[N] = <path>` — MUSIC2's wnoise reader expects strict GRAFIC
Fortran-unformatted records, not the plain dump format MUSIC2 writes by default
(it will throw "corrupt random number file").

Reproducible pipeline: `tests/validation/matched_noise/run.sh` — produces
`~/Documents/music_validation/figures/matched_noise_<lpt>_l<lmin>-<lmax>_re<re>_b<box>_z<z>_s<sc>-<sf>.png`.
Flags: `--use-2lpt | --use-1lpt`, `--seed-coarse N --seed-fine N`,
`--boxlength MPC --zstart Z --levelmin L --levelmax L --ref-extent F`,
`--padding N --accuracy F --smooth N`.  Recommended-quality knobs:
`padding=16`, `accuracy=1e-9`, `smooth=5` — drops the m=16 interior Ψ
residual by ~30% at the cost of a moderately larger doubled-patch FFT.

**What the matched-noise test is — and is not.**  Zoom ICs exist to pump
compute into a specific region (halo, filament, void, lensing target) at much
higher resolution than the user could afford box-wide, while the rest of the
box supplies the long-range tides at coarse resolution.  The user gets the
correct large-scale environment for free, and the fine particles resolve the
small-scale physics that motivated the run.

The matched-noise test asks "would a hypothetical full-box unigrid at the
patch resolution have produced bit-identical noise here?"  That is an
internal consistency check on the zoom machinery — not the user-facing goal.
What the user actually needs is **statistical equivalence**: ICs in the patch
with the right P(k), the right local tidal tensor, and the right cosmic
environment, so the halo that forms is physically the same one a
(computationally impossible) full-resolution box would produce.

A residual of ~10⁻²σ on the displacement field — concentrated at the patch
face, decaying inward with a smooth k-space spectrum — does not visibly
disturb any of that.  The halo at the patch centre has the same mass,
profile, and accretion history.  The bit-equality target only becomes
load-bearing for code-validation paper figures, not for science runs.

### Corrfunc

`compute_xi` links against Corrfunc. Resolution order:
1. `CORRFUNCDIR` make variable: `make -C src CORRFUNCDIR=/path/to/Corrfunc`
2. `--corrfunc-dir` flag: `./run_pipeline.sh --corrfunc-dir /path/to/Corrfunc`
3. `../Corrfunc` — sibling of the repo root (default)
4. Clones from `https://github.com/manodeep/Corrfunc` into `../Corrfunc` and builds if not found

## Maintenance Rules

- **Always update `README.md` and `CLAUDE.md`** when making structural changes to the repo: new directories, moved or renamed files, added/removed tools. Update the directory layout block and any file references in the same commit as the structural change.
- **Always compile LaTeX from the `notes/` directory.** Run `make` (or `make -C notes`) rather than invoking `pdflatex` directly from the repo root — otherwise `.log`, `.out`, `.aux` files land in the root instead of `notes/`. The notes/Makefile enforces this via `cd $(NOTESDIR) &&` before every pdflatex call.
