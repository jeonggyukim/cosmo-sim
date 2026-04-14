# CLAUDE.md — MUSIC2 and monofonIC

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

## Velocity Potential φ_v and LPT Displacement/Velocity Generation

There is no variable literally named `phi_v` in MUSIC2. The velocity potential is named **`u`** (1LPT branch) or **`u1`** (2LPT branch) in `src/main.cc`. These correspond exactly to Jenkins (2010) φ⁽¹⁾ and φ⁽²⁾.

Reference: Jenkins (2010), MNRAS 403, 1859 — "Second-order Lagrangian perturbation theory initial conditions for resimulations." https://ui.adsabs.harvard.edu/abs/2010MNRAS.403.1859J

### 1LPT branch (`use_2LPT = no`)

| Step | Code (`src/main.cc`) | Physics |
|---|---|---|
| Velocity source | `GenerateDensityHierarchy(..., theta_cdm, ...)` line 831 | θ = −f H δ (velocity divergence TF) |
| Poisson solve | `the_poisson_solver->solve(f, u)` line 839 | ∇²φ⁽¹⁾ = θ |
| Velocity | `gradient(icoord, u, data) *= cosmo_vfact` lines 858, 861 | **v = f₁ a H/h × (−∇φ⁽¹⁾)** |
| Displacement source | `GenerateDensityHierarchy(..., delta_cdm, ...)` line 625 | δ (density TF) |
| Poisson solve | `the_poisson_solver->solve(f, u)` line 637 | ∇²φ⁽¹⁾ = δ |
| Displacement | `gradient(icoord, u, data)` line 663 | **Δx = −∇φ⁽¹⁾** |

Velocity uses `theta_cdm`, displacement uses `delta_cdm`. These differ only by the growth rate factor absorbed into `cosmo_vfact = f × a × H(a)/h` (`src/cosmology_calculator.hh` line 346).

### 2LPT branch (`use_2LPT = yes`)

**Velocities** (lines 981–1063):
1. `solve(f, u1)` → ∇²φ⁽¹⁾ = δ⁽¹⁾
2. `compute_2LPT_source(u1, f2LPT)` → S = Σᵢ<ⱼ (φ,ᵢᵢ φ,ⱼⱼ − φ,ᵢⱼ²)  [Jenkins eq. 5; Hessian determinant terms]
3. `solve(f2LPT, u2LPT)` → ∇²φ⁽²⁾ = S
4. `u2LPT *= 6.0/7.0; u1 += u2LPT` → φ_vel = φ⁽¹⁾ + (6/7)φ⁽²⁾
5. `gradient(icoord, u1, data) *= cosmo_vfact` → **v = f₁aH/h × (−∇φ_vel)**

**Displacements** (lines 1163–1229):
1. Re-solve u1 from delta_cdm, re-solve u2LPT from 2LPT source
2. `u2LPT *= 3.0/7.0; u1 += u2LPT` → φ_disp = φ⁽¹⁾ + (3/7)φ⁽²⁾
3. `gradient(icoord, u1, data)` → **Δx = −∇φ_disp**

The **6/7 vs 3/7** factors reflect D₂/D₁² ≈ −3/7 (Einstein–de Sitter); velocities pick up an extra factor of 2 from dD₂/dt vs dD₁/dt.

### Gradient method: finite differences

`the_poisson_solver->gradient()` (`src/poisson.cc` lines 156–179) uses **configurable-order finite differences** (2nd, 4th, or 6th order stencils from `src/fd_schemes.hh`) — not ik multiplication in Fourier space.

### Jenkins (2010) formula: D·v − ½D²(dv/dD)

Jenkins (2010) describes an alternative approach to recover 2LPT displacements from the velocity field via a Taylor expansion in D, without re-running the Poisson solver. MUSIC2 does **not** use this — it takes the direct route of solving two Poisson equations for φ⁽¹⁾ and φ⁽²⁾. The Jenkins formula is mainly useful in resimulation contexts where you only have particle snapshots.

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

---

## monofonIC: Higher-Order ICs for Two-Fluid (Baryon+CDM) Simulations

Source: `~/Dropbox/Projects/monofonIC`

References:
- Hahn, Rampf & Uhlemann (2021), MNRAS 503, 426 — "Higher order initial conditions for mixed baryon–CDM simulations" https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..426H
- Rampf, Uhlemann & Hahn (2021) — companion theory paper

monofonIC is the successor to MUSIC2. The key advances over MUSIC2 (and over Jenkins 2010) are: **3LPT support**, a novel **Propagator Perturbation Theory (PPT)** method, and proper treatment of **CDM and baryons as two distinct fluids**.

### What monofonIC adds over MUSIC2 / Jenkins (2010)

| Feature | Jenkins (2010) | MUSIC2 | monofonIC / Hahn (2021) |
|---|---|---|---|
| LPT order | 2LPT | 2LPT | 3LPT + PPT |
| Fluid model | single | single (CDM≈matter) | two-fluid: CDM + baryons separately |
| Gradient method | finite differences | finite differences (2nd/4th/6th order) | ik multiplication in Fourier space (exact) |
| Aliasing suppression | none | none | Orszag 3/2-rule padding for LPT convolutions |
| Fixed amplitude | no | `fix_mode_amplitude = yes` | `DoFixing = yes` (Angulo & Pontzen 2016) |
| Paired ICs | no | no | `DoInversion = yes` |
| Particle lattice | SC | SC | SC, BCC, FCC, glass |
| Reference redshift | z=0 backscaled | `ztarget=0` backscaled | z_ref ≈ 2.125 (more accurate TF) |
| Transversal (vector) 3LPT | no | no | yes (A3 field) |

### Two-fluid framework (Hahn 2021 §2)

Hahn (2021) reformulates the CDM+baryon system in terms of **sum** (m) and **difference** (bc) variables (eq. 3):

```
δ_m = f_b δ_b + f_c δ_c     (total matter = baryon fraction × δ_b + CDM fraction × δ_c)
δ_bc = δ_b − δ_c             (baryon–CDM difference field)
```

At linear order δ_bc ≠ 0 due to baryon acoustic oscillations and Jeans damping. The combined displacement ξ^m drives gravity (Poisson source); the difference field δ_bc introduces a separate potential that separates baryons from CDM.

MUSIC2's approach of using `delta_baryon` vs `delta_cdm` transfer functions is only first-order accurate for the two-fluid case. monofonIC evolves both fluids self-consistently to 2LPT/3LPT.

### LPT potentials in monofonIC (`src/ic_generator.cc`)

| Variable | Order | Jenkins analogue | Code location |
|---|---|---|---|
| `phi` | 1LPT, φ⁽¹⁾ | φ⁽¹⁾ = u1 | lines 433–501 |
| `phi2` | 2LPT, φ⁽²⁾ | φ⁽²⁾ = u2LPT | lines 508–542 |
| `phi3a` | 3LPT scalar (a) | — | lines 545–575 |
| `phi3b` | 3LPT scalar (b), depends on φ⁽²⁾ | — | lines 576–597 |
| `A3x/y/z` | 3LPT transversal vector | — | lines 580–597 |

φ⁽¹⁾ is built **directly in Fourier space** (not via a real-space Poisson solve):
```cpp
phi(k) = white_noise(k) × sqrt(P(k)) / k²   // = δ(k) / k²
```

The 2LPT source is the Hessian determinant sum (Hahn 2021 eq. 20 / Jenkins eq. 5), computed using `OrszagConvolver` with 3/2-padded grids to suppress aliasing:
```cpp
Conv.convolve_SumOfHessians(phi, {0,0}, phi, {1,1}, {2,2}, assign_to(phi2));  // φ,00(φ,11+φ,22)
Conv.convolve_Hessians(phi, {1,1}, phi, {2,2}, add_to(phi2));                 // φ,11 φ,22
Conv.convolve_Hessians(phi, {0,1}, phi, {0,1}, subtract_from(phi2));          // −φ,01²
// ... etc.
phi2.apply_InverseLaplacian();   // solve ∇²φ⁽²⁾ = source
```

### Growth factors and velocity scaling (`include/cosmology_calculator.hh`)

monofonIC solves a coupled ODE system for all growth factors simultaneously:

| Variable | ODE solution | Physical meaning |
|---|---|---|
| D (y[1]) | 1LPT growth factor | D₁(a) |
| E (y[3]) | 2LPT growth factor | D₂(a) ≈ −(3/7)D₁² |
| Fa (y[5]) | 3LPT scalar (a) | D₃ₐ(a) |
| Fb (y[7]) | 3LPT scalar (b) | D₃ᵦ(a) |
| Fc (y[9]) | 3LPT transversal | D₃꜀(a) |

Velocity factors `vfac1..vfac3c` = `Ḋₙ/Dₙ/h` (line 259–264), analogous to MUSIC2's `cosmo_vfact = f₁ × aH/h` but computed for each LPT order separately.

Displacements and velocities are combined as (lines 883–1007):
```
Δx_i = −∂_i (φ⁽¹⁾ + φ⁽²⁾ + φ⁽³ᵃ⁾ + φ⁽³ᵇ⁾) + curl term from A3
v_i  = −∂_i (vfac1·φ⁽¹⁾ + vfac2·φ⁽²⁾ + vfac3a·φ⁽³ᵃ⁾ + vfac3b·φ⁽³ᵇ⁾) + vfac3c × curl(A3)
```

where ∂_i = i·k_i in Fourier space (exact, no FD truncation error).

### Propagator Perturbation Theory (PPT) (Hahn 2021 §2.4)

PPT is a novel alternative to standard LPT, formulated as a Schrödinger-like equation:
```
iħ ∂_D ψ_α = −(ħ²/2) ∇²ψ_α + V_eff ψ_α
```
where the wavefunction ψ_α encodes the fluid displacement field, ħ is a free parameter (set by the grid spacing / Nyquist condition, eq. 26), and V_eff is an effective potential derived from φ⁽²⁾.

At leading order (free propagator, V_eff = 0): recovers Zel'dovich (1LPT).
At next-to-leading order (2PPT, eq. 20): V_eff = (3/4)(φ⁽²⁾_,ii − φ⁽¹⁾_,ij φ⁽¹⁾_,ij)

PPT is implemented as a sequence of **drift** (Fourier-space, multiplication by e^{−iħk²ΔD/2}) and **kick** (real-space, multiplication by e^{−iΔD·V_eff/ħ}) operators, executed as a leapfrog (eq. 25):
```
ψ(x; a) = DFT⁻¹ { e^{−iħk²/2} DFT{ e^{−iKick} ψ^ini } }   [schematically]
```

Key advantage of PPT over LPT: preserves Hamiltonian structure, so no spurious high-order modes are excited. The baryon power spectrum in PPT agrees with full Eulerian simulation to sub-percent level at z ≲ 24.

### Backscaling reference redshift

monofonIC uses z_ref = 2.125 (not z=0) as the reference for the CLASS transfer function (Hahn 2021 §2.5). At z=0, decaying modes and relativistic effects have already been erased; backscaling from z_ref ≈ 2 captures the BAO scale and the dominant large-scale modes more accurately. MUSIC2 uses `ztarget = 0` (default).
