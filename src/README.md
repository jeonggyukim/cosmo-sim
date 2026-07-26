# `src/` — C estimator (ξ, ψ) for SWIFT ICs

`compute_xi` estimates the matter two-point correlation function ξ(r) —
and, with `--vel`, the velocity correlation ψ(r) = ⟨v·v'⟩ — from a
SWIFT IC HDF5 file.  It writes ASCII tables and produces no plots (use
`plot-ic` from `icpipe` for that).

## Build

```bash
make -C src                          # → bin/compute_xi (HDF5 + FFTW only)
make -C src clean
```

Defaults:
- macOS: clang + `-Xclang -fopenmp` + Homebrew `libomp` (keg-only).
  HDF5 and FFTW come from the regular `/opt/homebrew/{include,lib}`
  symlinks.
- Linux clusters: gcc + plain `-fopenmp`; paths default to the user's
  `$HOME/local/...` or `$HOME/libs/<arch>_gnu/...` per
  hostname-specific branch in the Makefile.

## `compute_xi` — CIC-grid autocorrelation

Deposits the N particles onto an Ngrid³ CIC density grid, normalises
to `den = 1 + δ`, and autocorrelates the grid.  By the
Wiener–Khinchin theorem this equals an FFT P(k) → Hankel transform →
ξ(r), and that is in fact the default path (`--fft`).  A direct
real-space lag sum is also available (`--no-fft`).

Estimator (Landy–Szalay over cells):

```
ξ(r) = (DD − 2 DR + RR) / RR  =  ⟨δ(x) · δ(x+r)⟩
```

```
./bin/compute_xi --input <ics.hdf5> [options]
```

Key options (see `--help` for the full set):

- `--Ngrid N` — CIC grid resolution (default: `cbrt(N_particles)`).
- `--assignment ngp|cic|tsc|pcs` — ξ/ψ grid mass assignment (default
  **cic**).  The ξ/ψ autocorrelation is not window-deconvolved, so a
  higher order only widens the real-space smoothing kernel — PCS's
  near-Nyquist anti-aliasing advantage applies to P(k), not the
  correlation functions.  The optional `--pk` output uses its own
  `--pk-assignment` (default **pcs**, interlaced + W² deconvolution).
- `--nbins / --rmin / --rmax` — radial binning (Mpc); defaults are
  cell-size and BoxSize/2.
- `--mode 3d | 1d-x | 1d-y | 1d-z` — full 3D vs single-axis projection.
- `--periodic xyz | xy | x | ...` — wrap directions; `--no-fft`
  forced if not fully periodic.
- `--vel` — also compute ψ(r) = ⟨v·v'⟩ from the CIC velocity grid;
  written to `--vel-output FILE`.
- `--output FILE` — main ξ table (columns:
  `r_avg r_low r_high xi DD DR RR`).  ψ is `r_avg r_low r_high psi
  [(km/s)²]`.

## Why a density grid rather than particle pair counting

This tool works at any z, including z = 200.  A pair-counting estimator
does not, and the reason is worth stating.

Pair counting counts pairs of actual particles, so its shot noise is
Poisson, σ_ξ ≈ √(1+ξ)/√N_pairs.  That is fine at low z, where the
clustering signal is large.  At z ≳ 10 the matter signal is suppressed
by D(z)² ≈ 1/(1+z)² to ~10⁻⁶ and drowns in that noise.

Lattice ICs escape this because their unperturbed particle positions
form a regular grid, which makes the shot noise sub-Poissonian:
σ_δ ≈ D(z)·σ₀ ≪ 1.  The CIC autocorrelation inherits that, so the noise
on ξ falls with D(z) alongside the signal and the tiny high-z
cosmological signal stays visible.  See the top-of-file block in
`compute_xi.c` for the estimator derivation.

## Files

```
src/
├── Makefile     # `make -C src`
└── compute_xi.c # ~1000 lines, self-contained CIC + FFT estimator
```

The binary lands in `../bin/` (gitignored).  `run-pipeline` calls
`./bin/compute_xi --vel` as step 6.
