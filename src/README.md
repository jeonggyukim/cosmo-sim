# `src/` — C estimators (ξ, ψ) for SWIFT ICs

Two C programs that estimate the matter two-point correlation function
ξ(r) — and, for `compute_xi_cic --vel`, the velocity correlation
ψ(r) = ⟨v·v'⟩ — from a SWIFT IC HDF5 file.  Both write ASCII tables;
neither produces plots (use `plot-ic` from `icpipe` for that).

## Build

```bash
make -C src                          # → bin/compute_xi, bin/compute_xi_cic
make -C src CORRFUNCDIR=/path/to/Corrfunc
make -C src clean
```

Defaults:
- `CORRFUNCDIR` → `../Corrfunc` (sibling of repo root); cloned + built
  if missing.  Override with the make variable or
  `run_pipeline.sh --corrfunc-dir`.
- macOS: clang + `-Xclang -fopenmp` + Homebrew `libomp` (keg-only).
  HDF5 and FFTW come from the regular `/opt/homebrew/{include,lib}`
  symlinks.
- Linux clusters: gcc + plain `-fopenmp`; paths default to the user's
  `$HOME/local/...` or `$HOME/libs/<arch>_gnu/...` per
  hostname-specific branch in the Makefile.

## `compute_xi` — particle-pair counting (Corrfunc)

Counts pairs of *actual particles* falling in each (rmin, rmax) bin and
applies the Peebles–Hauser estimator for a periodic box:

```
ξ(r) = DD(r) / DD_rand(r) − 1
```

RR (random–random) is analytic for a periodic box, so no random
catalogue is needed.  Corrfunc's grid acceleration brings the cost
from O(N²) down to O(N × N_in_shell).

```
./bin/compute_xi <ics.hdf5> <binfile> [nthreads] [-n NSUB] [-s SEED]
```

- `binfile` — two columns per line: `rmin rmax` (Mpc).  Generate with
  `make-rbins --hdf5 file.hdf5` (rmin = 2 × mean spacing, rmax = L/3).
- `-n NSUB` — subsample to NSUB particles before pair counting; if
  omitted, NSUB is auto-chosen for runtime.
- `-s SEED` — subsample random seed (default 42).

Output to stdout: `r_avg r_low r_high xi npairs`.  Redirect with `>`.

**Where it works**: low z, where the cosmological signal is well above
the Poisson particle shot noise σ_ξ ≈ √(1+ξ)/√N_pairs.

**Where it fails**: z ≳ 10.  At high z the matter signal is suppressed
by D(z)² ≈ 1/(1+z)² to ~10⁻⁶, drowning in shot noise.  Use
`compute_xi_cic` instead.

## `compute_xi_cic` — CIC-grid autocorrelation

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
./bin/compute_xi_cic --input <ics.hdf5> [options]
```

Key options (see `--help` for the full set):

- `--Ngrid N` — CIC grid resolution (default: `cbrt(N_particles)`).
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

**Where it works**: any z, including z = 200.  Lattice ICs have
sub-Poissonian shot noise σ_δ ≈ D(z)·σ₀ ≪ 1 because the unperturbed
particle positions form a regular grid; the CIC autocorrelation
inherits that, so the tiny high-z cosmological signal stays visible.

## Choosing between them

| IC redshift  | use                       | why                                    |
|--------------|---------------------------|----------------------------------------|
| z ≲ 5        | `compute_xi`              | strong signal, faster, no Ngrid tuning |
| 5 ≲ z ≲ 10   | either                    | both work; CIC is cleaner              |
| z ≳ 10       | `compute_xi_cic`          | particle shot noise drowns the signal  |
| any z, velocity ψ(r) | `compute_xi_cic --vel` | only CIC estimator handles velocities  |

See CLAUDE.md "When ξ(r) is the wrong tool" for the full discussion
and `compute_xi_cic.c`'s top-of-file block for the estimator derivation.

## Files

```
src/
├── Makefile          # invoked as `make -C src` from the repo root
├── compute_xi.c      # ~300 lines, includes Corrfunc public headers
└── compute_xi_cic.c  # ~1000 lines, self-contained CIC + FFT estimator
```

Binaries land in `../bin/` (gitignored).  `run_pipeline.sh` calls
`./bin/compute_xi` (step 7) and `./bin/compute_xi_cic --vel` (step 8).
