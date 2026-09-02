# CLAUDE.md — cosmo-sim

This repository holds two things:

1. **An IC-generation pipeline.** Cosmological initial conditions from MUSIC2 or
   MUSIC2-monofonIC, validated against CLASS linear theory by measuring P(k),
   ξ(r), and ψ(r). Code in `icpipe/`, `src/`, `tools/`.
2. **A collection of LaTeX study notes** on cosmological ICs and field
   statistics, in `notes/`. Treat them as the second purpose of the repo, not
   an appendix to the first. The dark-matter physics notes (SIDM, fuzzy dark
   matter, field theory, dark-matter chemistry, classical mechanics) moved to
   the `ai-notes` repository on 2026-07-28; they had no dependency on this
   code.

## Directory Structure

```
cosmo-sim/
  src/            — compute_xi.c + Makefile (CIC-grid ξ/ψ estimator)
  bin/            — compiled C binaries (gitignored)
  icpipe/         — installable Python library (ICField, LinearTheory, io,
                    pipeline.py = end-to-end orchestration)
  icpipe/cli/     — console_scripts: run-pipeline, make-music-conf,
                    make-monofonic-conf, compute-pk, compute-pv, plot-ic
  scripts/analysis/ — ad-hoc CLIs (check_ic, plot_box_size_comparison,
                    plot_dr_histogram, xi_sweep)
  scripts/ic_search/ — pencil-subvolume P(k) and the seed sweep; needs the
                    monofonIC `lagrangian-density` fork and reads/writes runs
                    outside the repo (see its README)
  tools/          — build-music.sh, build-monofonic.sh, clean.sh, sbatch templates
  tests/          — bundled validation tests (matched-noise zoom)
  notebooks/      — worked examples using icpipe
  conf/           — MUSIC2 / monofonIC configs and CLASS ini files
  data/           — IC HDF5, CLASS P(k), wnoise binaries, measured tables
  plots/          — output figures
  notes/          — LaTeX study notes (see below)
  docs-claude/    — planning docs and the MUSIC2 reference
  music_build/, monofonic_build/ — compiled IC-code binaries (gitignored)
```

`data/*.hdf5` and `data/*.bin` are gitignored.

## Python Environment

Use `conda run -n cosmo python ...` for all Python in this project.

## Pipeline

```bash
conda run -n cosmo run-pipeline                        # MUSIC (default), N=256, L=1000, z=200
conda run -n cosmo run-pipeline --ic-code monofonic    # monofonIC (PLT, 3LPT)
conda run -n cosmo run-pipeline --ngrid 256 --lbox 512 --zstart 200
conda run -n cosmo run-pipeline --ic-source-dir /path/to/MUSIC2
```

Steps, each skipped if its output already exists:

1. Build the IC code
2. Build `compute_xi`
3. Generate the IC config from a template
4. Run the IC code → IC HDF5 + a CLASS ini
5. Run CLASS → `data/class_pk_z{z}_pk.dat`
6. Measure ξ(r) and ψ(r) (`compute_xi --vel`)
7. Measure P(k) (`compute-pk`)
8. Plot diagnostics (`plot-ic`) → `plots/pk_{stem}.png`

Remove outputs with `tools/clean.sh` (`--all` also removes IC HDF5 files).

**Two IC codes coexist** at the same N, z, L: monofonIC outputs carry a `_mono`
tag in every stem. Both write a CLASS ini during the run, which the driver
normalizes to `conf/input_class_parameters_{stem}.ini` and adapts for the
matter-P(k) run. `data/class_pk_z0_pk.dat` is cosmology-keyed and reused across
runs. monofonIC links CLASS as a library and builds no standalone `class`
binary, so a monofonic-only setup needs MUSIC's CLASS binary or a prebuilt
`class_pk`, or the theory overlay is skipped.

**Thread count**: `run-pipeline` defaults `--nthreads` to `os.cpu_count()`.
Override to match a cluster allocation.

**Cluster**: `--launcher "srun"` or `--mpi-ranks N` launch the IC step across
ranks; the orchestrator itself stays single-process.

## Key Files

**Drivers and builds**
- `run-pipeline` → `icpipe/cli/run_pipeline.py`, a thin CLI over
  `icpipe/pipeline.py`. `--ic-code {music,monofonic}` branches the front-end
  (build → config → run); everything downstream is code-agnostic since both
  write SWIFT HDF5. Preferred entry point.
- `tools/build-music.sh` — source resolution: `$MUSIC2_SOURCE_DIR`, `../MUSIC2`,
  else clone. macOS sets `FC=gfortran-14` and uses Homebrew; non-Darwin loads
  cluster modules. Skips if `music_build/MUSIC` exists.
- `tools/build-monofonic.sh` — builds into `monofonic_build/` with
  `-DENABLE_PLT=ON` (monofonIC defaults it off) and `-DENABLE_MPI=ON` (required
  to compile; the binary still runs single-rank). Source resolution:
  `$MONOFONIC_SOURCE_DIR`, `../monofonIC`, else clone. CLASS is fetched by its
  CMake.

**Configs** — `conf/CV_22_MUSIC_template.conf` and
`conf/CV_22_monofonIC_template.conf` share the CV_22 cosmology;
`make-music-conf` and `make-monofonic-conf` fill in N, z, L and write to
`conf/`.

**Python package `icpipe`** (`pip install -e .`)
- `ICField(hdf5, ngrid=..., interlace=True, h=..., load_velocities=True)` —
  loads positions/velocities, caches CIC density/momentum/velocity Fourier
  fields, `.power(field, ...)` for δ or velocity spectra.
- `LinearTheory.from_class(class_pk_dat, z=..., h=..., Omega_m=...)` — `Pk`,
  `Pv`, `xi`, `psi` from one CLASS table, units consistent across all four.
- `icpipe.io` — `read_pk`/`read_pv`/`write_pk`/`write_pv`, `read_wnoise`,
  `read_swift_ics`, `print_swift_ics_summary`.
- Tests: `pytest icpipe/tests/` (17 tests).

**Measurement and plotting**
- `compute-pk` — P(k) via mass assignment + FFT. `--assignment {ngp,cic,tsc,pcs}`,
  default pcs (most accurate near Nyquist).
- `compute-pv` — velocity power spectrum via CIC velocity assignment + FFT.
- `src/compute_xi.c` → `bin/compute_xi` — ξ(r) and ψ(r) by CIC density/velocity
  grid autocorrelation. Needs only HDF5 + FFTW; `make -C src`. `--vel` adds the
  velocity correlation. Works at any z, because lattice ICs have sub-Poissonian
  shot noise (σ_δ ≈ D(z)·σ₀ ≪ 1) — particle pair counting cannot do this, since
  at z ≳ 10 the signal is ~10⁻⁶ and drowns in Poisson noise.
- `plot-ic` — reads `pk_*.txt`, auto-detects `xi_cic_*.txt` and `vel_cic_*.txt`
  alongside it, overlays CLASS theory P(k), ξ(r), ψ(r).
- `scripts/analysis/check_ic.py` — verifies DC (k=0) modes of
  displacement/velocity; prints mass resolution and recommended SWIFT softening
  (ε = Δx/40…Δx/25). `--explain` for the preamble, `--hist PNG` for histograms.

## Units and conventions

- **SWIFT stores coordinates and BoxSize in Mpc, not Mpc/h.** All internal
  calculations use Mpc; convert with `h = H0/100` for plots.
- **Velocities are peculiar velocities in km/s.** MUSIC2's SWIFT plugin writes
  `v_out = a·H(a)·f·Ψ_comoving = v_pec`. SWIFT converts to its internal
  canonical momentum `u = a·v_pec` only at IC read time. So ψ(r) from
  `compute_xi` is already in (km/s)² and compares directly to
  ψ_theory(r) = [H(z)f(z)]²/(2π²) ∫ P(k) j₀(kr) dk — no a² rescaling.
- **MUSIC2 writes `wnoise_NNNN.bin` and `input_class_parameters.ini` to the
  CWD** (hardcoded). `run-pipeline` moves both; direct invocations need
  `mv wnoise_*.bin data/ && mv input_class_parameters.ini conf/`.
- **Shot noise limits the k-range**: measurable while P(k,z) > P_shot = V/N × h³.
  At high z, P(k) is suppressed by D(z)² ≈ 1/(1+z)², so P_shot wins at lower k
  for larger or coarser boxes. At z=45: 256³/L=500 is shot-noise dominated
  throughout; 512³/L=500 gives k ≲ 0.2; 1024³/L=500 gives k ≲ 0.7; 256³/L=25
  gives k ≲ 3 h/Mpc.

## Notes

LaTeX write-ups in `notes/`; full list and reading order in `notes/README.md`.

| File | Contents |
|------|----------|
| `cosmo_ic.tex` | Cosmological ICs: fluid equations, ZA, 2LPT, IC generation, starting redshifts, P(k)/ξ(r)/ψ(r), one-loop P(k); §11 MUSIC2 / monofonIC implementation. App. A Fourier conventions, App. B P(k) estimation |
| `ic_sampling.tex` | P-sampled vs ξ-sampled ICs (Pen 1997, Sirko 2005, Hahn & Abel 2011); box window truncation |
| `restriction_lpt.tex` | Restriction of δ and Ψ⁽¹⁾; why restricting the displacement potential is Poisson-inconsistent |
| `music2_internals.tex` | File-and-line walkthrough of MUSIC2's δ construction and zoom noise hierarchy |
| `cosmo_stat.tex` | Statistics of cosmological fields: ensembles, ergodicity, ξ(r), P(k), covariance, Gaussian random fields, generating and measuring them |
| `ic_search.tex` | Why adopting the seed whose subvolume P(k) best matches linear theory selects an atypical region. The estimator is unbiased for theory convolved with the subvolume window, 34% below theory at the fundamental, so matching the raw theory needs an upward fluctuation. Selection displaces the subvolume's large-scale power by 1.26σ and every other property by its correlation with that power times 1.26; the displacement saturates, so a million-seed search is no worse than a thousand. 83,542 realisations at 128³. Controls: random criterion 0.000, convolved-theory criterion 0.016. Written in journal register, not the textbook register of the other notes |
| `xi_estimators.tex` | Why a direct lag sum and a zero-padded FFT are the same estimator: grid Landy–Szalay in the infinite-random limit and Slepian & Eisenstein's N = D−R convolution form, Wiener–Khinchin as a finite identity, circular vs linear autocorrelation and the padding condition d ≤ P−N, the separable N−d pair count, per-axis boundaries for pencil beams, cost scaling, and the gridding artefacts the FFT route concedes |

Planning docs in `docs-claude/`: `CLAUDE_MUSIC2.md` (MUSIC2 code structure,
transfer functions, file formats), `baryon_ic_plan.md`, `swift_gpu_gravity_plan.md`
(KIAS GPU-porting project), plus per-feature plans.

## Pencil subvolumes (`scripts/ic_search/`)

Measures P(k) in a pencil subvolume — 1/8 of the box in two axes, the full box in
the third — of a monofonIC δ(q) field, and sweeps seeds looking for a pencil that
matches linear theory.

The one fact that governs everything there: masking multiplies in configuration
space, so ξ divides out the mask autocorrelation exactly, but P(k) **convolves**
and cannot be deconvolved per mode. A pencil measures a spectrum 34% below theory
at the fundamental, deterministically. Convolving the theory with the pencil
window reproduces the measurement to 0.4%, so comparisons are made in the observed
basis and never deconvolved. Selecting a seed on agreement with the *unconvolved*
theory selects a realisation whose scatter cancels that geometric deficit —
`notes/ic_search.tex` is the same argument, measured for P(k).

Requires the `lagrangian-density` branch of the monofonIC fork
(`github.com/jeonggyukim/monofonIC`), which adds `[setup] LagrangianDensityOnly`.
Runs live outside the repo under `$MONOFONIC_TESTS` (default
`~/Documents/monofonic-tests`); `scripts/ic_search/paths.py` holds every path.
`DoFixing = no` is the production setting.

```bash
make -C notes              # regenerate figures + compile all PDFs
make -C notes figures      # figures only
make -C notes notes        # PDFs only
make -C notes clean
```

### Prose in `.tex` and `.md` files

The rules are in `~/.claude/CLAUDE.md` under `## Prose`, and they apply to
every word written into a `.tex` or `.md` file here. They are not repeated in
this file. The short form: write the plainest sentence that states the fact,
then stop.

Defects that reached `notes/cosmo_stat.tex` and had to be corrected sentence
by sentence, kept here as worked examples: "the bridge between theory and
data", "for free", "the whole game", "the price of a finite box", "the rescue
is ergodicity", "through the lens of ordinary statistics", "each cell knows
nothing about its neighbours", "the right-hand side mentions x nowhere",
"even though it looked as though it might".

The money-and-injury family for "a method removes something" — cost, price,
buys, pays, damage, penalty, free — recurred six times in a first draft of
`notes/ic_search.tex` after being corrected once in `cosmo_stat.tex`.
Name the quantity: "removes 21% of the scatter", not "costs a fifth of the
scatter".

### Writing style for the notes

The rule above applies first. What follows is specific to `notes/`.

These notes are for self-study. The reader is meeting the material for the first
time, or returning to it after a long gap, and has nothing else in front of them.
Write the way a good textbook written for international students reads:

- Say what you are about to do before doing it.
- Justify each non-obvious step. If a line of algebra holds because mixed partial
  derivatives commute, say so.
- Define every term, symbol, and name where it first appears, including where a
  name comes from when that makes it memorable.
- State in words what a result means once it has been derived.
- Give a concrete worked example where a general statement is hard to picture.
- One idea per sentence; plain word order; no inverted or fronted clauses.

Length is not a cost to this reader. Having to re-read a paragraph three times
is. A rewrite that triples the length of a compressed passage is the correct
trade. The opening sections of `notes/kinetic_theory.tex` in the `ai-notes`
repository (formerly §A.1 and §A.2 of `sidm.tex`) are the reference for this
style.

This is the opposite of the standard for a PR body, where the reader already
knows the material and padding is a defect. Do not mix the two.

## Matched-noise validation (MUSIC2-anisotropic-zoom fork)

For the Hahn 2011 §4.3 zoom-vs-unigrid check in the `MUSIC2-anisotropic-zoom`
fork, the working combination is `[random]/kaveraging = no` (makes level-N white
noise coordinate-deterministic, per-cell RNG keyed on `(seed[N], i, j, k)`, with
no Meyer-window FFT splice across array shapes) and `[setup]/density_boundary =
yes` (Hahn 2011 §2.3.3 three-term density assembly at the coarse-fine boundary).
With both set and the same `seed[levelmax]` in the zoom and the unigrid
(`levelmin=levelmax`) conf, the patch-interior δ(q) rms residual is 2.6×10⁻⁴σ at
margin = 24 (commit `b90cfad`).

Do not pass the zoom's `wnoise_NNNN.bin` as `seed[N] = <path>` — MUSIC2's wnoise
reader expects strict GRAFIC Fortran-unformatted records, not the plain dump
format MUSIC2 writes by default, and throws "corrupt random number file".

Reproducible pipeline: `tests/validation/matched_noise/run.sh`. Recommended
quality knobs: `padding=16`, `accuracy=1e-9`, `smooth=5` — drops the m=16
interior Ψ residual by ~30% for a moderately larger doubled-patch FFT.

Note what this test does and does not show. It asks whether a hypothetical
full-box unigrid at the patch resolution would have produced identical noise —
an internal consistency check on the zoom machinery. What a science run needs is
statistical equivalence: the right P(k), local tidal tensor, and cosmic
environment in the patch, so the halo that forms matches what a full-resolution
box would give. A ~10⁻²σ displacement residual at the patch face, decaying
inward, does not disturb that. Bit-equality matters for code-validation figures,
not for science runs.

## Subagent and skill guidance

- Run `/prose-review` on any prose document before the user reads it. It routes
  by audience: `prose-reviewer` for the LaTeX notes and docs (reports passages
  too compressed to follow), `pr-body-reviewer` for PR bodies and issue text
  (reports missing evidence and padding). Shared word and sentence rules:
  `~/.claude/skills/prose-review/register.md`.
- Use the **Explore** agent for searches across `icpipe/`, `src/`, `scripts/`.

## Maintenance Rules

- **Update `README.md` and `CLAUDE.md`** in the same commit as any structural
  change: new directories, moved or renamed files, added or removed tools.
- **Compile LaTeX with `make -C notes`**, never `pdflatex` from the repo root —
  otherwise `.log`, `.aux`, `.out` land in the root. The notes/Makefile enforces
  this with `cd $(NOTESDIR) &&` before every pdflatex call.
