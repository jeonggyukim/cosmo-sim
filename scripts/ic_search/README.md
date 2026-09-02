# `scripts/ic_search/` — pencil-subvolume P(k) and the seed search

Measures P(k) inside a **pencil** subvolume of a periodic box — one eighth of the
box in two axes, the full box in the third — and asks how many random seeds must
be tried before some pencil matches linear theory.

The short answer, from 11,520 pencils: a pencil measures a spectrum that is
**34 % low at the fundamental**, and that deficit is a property of the geometry,
not of the realisation. Convolving the theory with the pencil window reproduces
the measurement to 0.4 %. Selecting a seed on agreement with the *unconvolved*
theory selects a realisation whose scatter happens to cancel that deficit.

## Why the window cannot be divided out

Masking is a product in configuration space. For ξ(r) an autocorrelation
factorises lag by lag, so the mask contributes one number per lag,
`n(d) = Σ_x W_x W_{x+d}`, which divides out exactly. In Fourier space the same
product is a **convolution**,

```
<|F(k)|²> ∝ Σ_k' |Ŵ(k−k')|² P(k')
```

so there is no per-mode number to divide by. These scripts therefore never
deconvolve: they push the theory forward through the same window and compare in
the observed basis. This is Park, Vogeley, Geller & Huchra (1994), eqs. (9)–(13),
except that a separable rectangular window makes their eq. (11) one FFT rather
than a suite of mock surveys, which is why the residual here is 0.4 % instead of
their < 20 %.

For a pencil the mask is separable, `W = W_x W_y · 1`, so the long axis
contributes a delta function at `k_∥ = 0`: **the convolution acts only in the
transverse plane and there is no mixing along the line of sight.** The transverse
window has its first zero at `Δk⊥ = 2π/ℓ⊥`, eight fundamental modes for a
one-eighth pencil.

## Requirements

* Python with `numpy`, `h5py`, `matplotlib`; `icpipe` (this repo) for the
  mass-assignment kernels used by the particle cross-checks.
* monofonIC built from the `lagrangian-density` branch of
  <https://github.com/jeonggyukim/monofonIC>, which adds
  `[setup] LagrangianDensityOnly` — it writes δ(q) for total matter, CDM and
  baryons and returns before any LPT work, so a 64³ realisation takes ~1.5 s.

Paths are resolved in `paths.py` and overridden by environment variables, so
nothing outside that file is machine-specific:

| variable | meaning | default |
|---|---|---|
| `MONOFONIC_TESTS` | root holding `data/` and the reference run; figures are written here | `~/Documents/monofonic-tests` |
| `MONOFONIC_REF` | reference run supplying the config template and the CLASS table | `$MONOFONIC_TESTS/n64_deltaq_z200_L700` |
| `MONOFONIC_BIN` | the monofonIC binary | a path under `~/Library/CloudStorage/...` |

The reference run is one ordinary `LagrangianDensityOnly` run at N = 64,
L = 700 Mpc/h, z = 200. The sweep copies its config, substituting seed, grid,
box, `DoFixing` and output path.

## Pipeline

Run from this directory (the scripts import `paths` as a sibling module).

### 1. Generate and measure — `pencil_seed_sweep.py`

One monofonIC run per seed, then P(k) for the full box and for all 192 disjoint
pencils — 64 transverse tiles × 3 orientations — for all three species.

```bash
# production setting: amplitudes are not fixed
python pencil_seed_sweep.py --seed0 3001 --nseeds 30 \
       --out $MONOFONIC_TESTS/data/pencil_sweep_n64_L700_x30_nofix

# the fixed-amplitude comparison
python pencil_seed_sweep.py --seed0 2001 --nseeds 30 --dofixing yes \
       --out $MONOFONIC_TESTS/data/pencil_sweep_n64_L700_x30
```

`--keep-fields` retains each seed's δ(q) (6 MB per seed); by default it is
deleted after measurement, since the seed reproduces it exactly. About 4.8 s per
seed: ~1.5 s to generate the field, the rest for 579 FFTs.

The estimator, with `W` the 0/1 pencil mask and `f = <W²>` its volume fraction:

```
F(k)      = (1/N³) Σ_x W_x δ_x exp(−ik·x)
P_meas(k) = V |F(k)|² / f
```

No shot noise is subtracted — δ(q) is a continuum field on a grid, not a point
process — and no padding is applied, because padding is what makes the ξ route
exact and does nothing for P(k). The **global** mean is used, never the
subvolume's own; using the local mean would add the integral constraint on top
of the window, and that one cannot be cleanly undone.

**Output layout** (per sweep directory):

```
theory.hdf5            k[32], nmodes[32], P_theory[3,32], P_win[3,32]
seed_NNNNN/pk.hdf5     k[32], P_full[3,32], P_pencil[3,192,32],
                       pencil_axis[192], pencil_i[192], pencil_j[192]
seed_NNNNN/deltaq.conf, run.log, and monofonIC's own theory tables
summary.hdf5           one row per (seed, species, pencil) with rms and mean
                       log-ratio against both reference curves
```

The species axis is `['matter', 'cdm', 'baryon']`, recorded in the `species`
attribute. `pencil_axis` is the long axis (0 = x, 1 = y, 2 = z); `pencil_i` and
`pencil_j` index the two transverse axes in ascending order, each 0–7, so tile
`i` spans cells `[8i, 8i+8)`.

> **Units.** monofonIC tabulates `[A(k)·D₊]²` in internal amplitude units
> carrying `(2π)^-3/2` from `volfac`. The standard-convention P(k) is
> **(2π)³ times** the table. Every script here applies that factor.

### 2. Hit rates — `analyze_sweep.py`

```bash
python analyze_sweep.py --data DIR --species matter
```

Reports, for three k bands, the distribution of
`D = rms ln(P_pencil/P_ref)` and how many pencils fall below 2 %, 5 % and 10 %,
against both the raw theory and the window-convolved theory. The band choice
dominates every number: above `2Δk⊥` the window is a sub-percent effect and most
pencils match; below it, none do.

### 3. Figures — `plot_sweep_summary.py`, `plot_deviation_stats.py`

```bash
python plot_sweep_summary.py    --data DIR --species matter
python plot_deviation_stats.py  --data DIR --species matter
```

The first is a 2×2: full box on top, pencils below, P(k) left and ratio right,
individual realisations in thin grey, the mean in red, and the pencils that best
match raw theory highlighted. The second gives the deviation histograms per band
and the cumulative hit rate. Both write to `$MONOFONIC_TESTS` named after the
sweep directory and species.

## Validation scripts

Not part of the sweep; they establish that the pieces are right.

| script | what it checks |
|---|---|
| `pencil_pk.py` | the pencil estimator against the window-convolved theory: 0.9976 median, 0.4 % worst above 2Δk⊥ |
| `plot_deltaq_check.py` | δ(q) against the back-scaled CLASS spectrum, with a 2LPT particle IC from the same seed as an independent estimator |
| `plot_species_pk.py` | the three species side by side, and the relative mode δ_bc |
| `compare_particle_vs_deltaq.py` | grid vs particle spectra on identical k bins, and the rms displacement |

## Two results worth knowing before using this

**Amplitude fixing changes the box, not the search.** `--dofixing no` is the
default here because it is what is used in practice. With `DoFixing = yes` the
full-box P(k) is identical across seeds to machine precision — the seed sets
phases only. Turning it off makes the full box scatter by a factor of two at the
fundamental but leaves the pencil statistics unchanged (median D 0.2721 → 0.2703),
because a pencil already mixes many modes through the window.

**The white noise is resolution-independent at fixed box size.** The N-GenIC seed
table is filled in shells around the origin, so one seed at 64³ and at 128³ shares
every mode the two grids have in common — verified to a maximum fractional
difference of 2.2 × 10⁻¹³ over 250,046 modes. A promising seed at 64³ keeps its
large-scale structure at higher resolution.
