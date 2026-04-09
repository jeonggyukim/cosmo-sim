# CLAUDE.md — MUSIC2

MUSIC2 is a C++17 application for generating nested-grid initial conditions for cosmological zoom simulations, using Lagrangian Perturbation Theory (1LPT/2LPT), FFTW3-based Poisson solving, and a plugin architecture for output formats, transfer functions, and random number generators.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

On macOS, Apple Clang lacks OpenMP — use GNU compilers:
```bash
CC=gcc-13 CXX=g++-13 cmake ..
```

Hint library paths if needed:
```bash
FFTW3_ROOT=<path> HDF5_ROOT=<path> cmake ..
```

CMake options:
- `CODE_PRECISION`: FLOAT | DOUBLE (default) | LONGDOUBLE
- `ENABLE_PANPHASIA`: ON/OFF (requires Fortran compiler)
- `ENABLE_CLASS`: ON/OFF (Boltzmann code for power spectra)
- `CMAKE_BUILD_TYPE`: Release | Debug | RelWithDebInfo | DebugSanAdd | DebugSanUndef

## Run

```bash
./MUSIC ../ics_example.conf
```

## Key Dependencies

| Library | Use |
|---------|-----|
| FFTW3 | FFT-based Poisson solver |
| GSL | ODE integration, interpolation, linear algebra |
| OpenMP | Parallelization |
| HDF5 (optional) | Arepo, Swift, ENZO, Generic output |
| Fortran (optional) | PANPHASIA RNG |

## Directory Structure

```
src/
  main.cc                  # Entry point
  mesh.hh                  # Grid hierarchy and mesh data structures
  mg_solver.hh             # Multigrid (FAS) solver
  mg_operators.hh          # Restriction, prolongation, smoothing
  poisson.cc/.hh           # Poisson solver (FFT and multigrid)
  perturbation_theory.cc   # 1LPT and 2LPT implementations
  cosmology_calculator.hh  # Growth factors, power spectrum
  config_file.hh           # INI-style config parser
  math/                    # vec3, mat3, interpolation, ODE, special functions
  plugins/
    output_*.cc            # Output formats (Gadget, AREPO, Swift, RAMSES, ...)
    transfer_*.cc          # Transfer functions (CAMB, CLASS, Eisenstein&Hu, ...)
    random_*.cc            # RNGs (MUSIC, PANPHASIA)
    region_*.cc            # Refinement region generators (ellipsoid, convex hull)
ext/
  panphasia/               # PANPHASIA Fortran source
data/
  ExampleConfigs/          # Example configuration files
tools/                     # Utility tools (compute_ellipsoid, etc.)
```

## Code Conventions

- Classes: `snake_case` (e.g., `multigrid_poisson_plugin`)
- Functions: `snake_case`
- Member variables: `snake_case_` with trailing underscore
- Type alias for floating-point precision: `real_t` (set at compile time)
- Header guards: `#pragma once`
- Plugins register themselves via factory pattern; see `output.hh`, `transfer_function.hh`, `random.hh`, `region_generator.hh` for base classes

## Adding a Plugin

1. Create `src/plugins/<type>_<name>.cc`
2. Subclass the relevant abstract base class
3. Register with the factory at file scope
4. CMake picks up all `.cc` files in `src/` and `src/plugins/` automatically

## Notes

- v1 and v2 produce different random noise hierarchies — not backwards compatible
- PANPHASIA requires a Fortran compiler; disable with `-DENABLE_PANPHASIA=OFF`
- CI runs on Ubuntu via GitHub Actions (`.github/workflows/cmake-multi-platform.yml`)

## Divergence from Hahn & Abel (2011)

The original MUSIC paper's central innovation was a **real-space convolution kernel** (`T_R(r)`, eq. 3–5 in the paper) to handle zoom (nested) grids correctly. This avoids the **integral constraint** error inherent in k-space sampling, which forces P(k=0) = 0 and suppresses large-scale power.

**Current MUSIC2 no longer implements this.** The `kernel_real_cached` class and `TransferFunction_real` were entirely removed in commit `bb9c07c` (Feb 2023, "removed realspace kernel"). Only the k-space kernel (`kernel_k`) remains.

### Consequence: large-scale power suppression for small boxes

K-space sampling enforces P(k=0) = 0, offsetting the correlation function by:

$$\Delta\xi = -\frac{4\pi}{V} \int_0^{r_\text{box}} \xi(r)\, r^2\, dr$$

The paper (Fig. 2) shows this is significant for L ≲ 100 h⁻¹ Mpc. **For a 25 Mpc/h box (CV_22_MUSIC.conf), this suppression is expected to be measurable.**

This can be verified by measuring P(k) from `ics_swift.hdf5` and comparing against the CLASS prediction from `input_powerspec.txt`.

### Why was the real-space kernel removed?

Likely a combination of:
1. **Memory**: real-space kernel required a double-padded grid (2N)³ per refinement level — 512 MB per level for N=256, vs. zero extra memory for k-space
2. **Complexity**: required disk caching (`temp_kernel_level*.tmp`) and FFTLog integration
3. **Practical accuracy**: for large boxes (L ≳ 100 h⁻¹ Mpc), the suppression is negligible

## IC Generation Pipeline

### White noise field (`wnoise_NNNN.bin`)

Written by `src/plugins/random_music.cc`. An intermediate file storing the Gaussian random number grid at a given refinement level so it can be reused across runs (e.g. different resolution or cosmology).

Binary format:
- Header: 3 × `unsigned int` → nx, ny, nz
- Data: nx × ny × nz × `real_t` (float32 by default), i-major order (slice by slice)

Reuse in config:
```ini
[random]
cubesize = 256
rngfname = wnoise_0008.bin
```

### Transfer function convolution

**Entry point:** `src/densities.cc`
- `GenerateDensityUnigrid()` (line ~255) for uniform grids
- `GenerateDensityHierarchy()` (line ~311) for zoom grids

**Core convolution:** `src/convolution_kernel.cc`, `perform()` (line ~32):
1. Forward FFT white noise field (real → complex)
2. For each k-mode: compute `|k|`, call `pk->at_k()` → `T(k)`
3. Multiply: `output(k) = whitenoise(k) × T(k) × fftnorm`
4. Inverse FFT back to real space

**Transfer function value** (`src/transfer_function.hh:231`):
```cpp
return sqrtpnorm_ * pow(k, 0.5*nspec_) * ptf_->compute(k, type_);
// = sqrt(pnorm) × k^(n_s/2) × T_plugin(k)
```
This is `sqrt(P(k))` — the square root of the power spectrum.

### `ptf_->compute(k, type)` — plugin interface

`transfer_function` (in `src/transfer_function.hh:85`) is an abstract base class with:
```cpp
virtual double compute(double k, tf_type type) const = 0;
```

`tf_type` enum selects which field to return:
| type | meaning |
|------|---------|
| `delta_cdm` / `delta_matter` / `delta_baryon` | density TF at `ztarget` |
| `theta_cdm` / `theta_matter` / `theta_baryon` | velocity divergence TF at `ztarget` |
| `delta_cdm0`, `theta_cdm0`, ... | same but at `zstart` |
| `delta_bc` | baryon–CDM relative TF |

For the CLASS plugin (`src/plugins/transfer_CLASS.cc:279`), `compute()`:
1. Converts k units: `k *= h_`
2. Returns 0 outside `[kmin_, kmax_]`
3. Looks up the requested field via log-log spline interpolation (`interpolated_function_1d`) of the CLASS output table
4. Returns `val * tnorm_` (normalized so T(k→0) = 1)
