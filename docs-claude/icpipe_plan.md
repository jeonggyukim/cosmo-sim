# icpipe: a small Python package for IC analysis

Plan for a refactor that pulls the shared CIC + FFT + binning code out
of `compute_pk.py`, the planned `compute_pv.py`, and `plot_ic.py` into
a small Python package, and adds the velocity-power-spectrum
diagnostic on top.

Author: Claude (Anthropic), 2026-04-29.
Supersedes the standalone-script approach in `velocity_pk_plan.md`.

## Goals

1. **DRY**: one CIC assignment, one CIC window deconvolution, one
   radial binning routine, one HDF5 reader. Currently this code is
   duplicated (or about to be) across `compute_pk.py` and the planned
   `compute_pv.py`.
2. **Reusable**: classes/functions importable from any script or
   notebook, not just the CLI tools.
3. **Self-consistent linear-theory predictions**: $P_\delta$, $P_v$,
   $\xi$, $\psi$ all derived from the same CLASS $P(k)$ via a single
   `theory` module — no risk of one diagnostic using a different
   normalisation than another.
4. **Testable**: unit tests for the building blocks (CIC,
   deconvolution, binning, theory).

## Package structure

```
icpipe/
├── __init__.py        # exports ICField, theory, etc.
├── field.py           # ICField class: HDF5 → CIC grids → spectra
├── theory.py          # Linear-theory P_δ, P_v, ξ, ψ from CLASS table
├── windows.py         # CIC window + deconvolution + Jing alias
├── binning.py         # radial-bin + log-bin utilities
└── io.py              # write_table / read_table for pk_*, pv_*, etc.
```

Six small modules, each ≲ 200 lines.

## Class API sketch

```python
from icpipe import ICField

f = ICField('data/ics_swift_n256_z200_L500.hdf5', ngrid=512, interlace=True)

# Cached gridded fields (computed lazily on first access):
f.delta        # (Ngrid,)*3 overdensity
f.momentum     # (3, Ngrid, Ngrid, Ngrid) momentum density
f.velocity     # (3, Ngrid, Ngrid, Ngrid) velocity (= momentum/density)

# Power spectra (returns a result dict {k, P, Nmodes, Pshot, ...}):
pk = f.power('delta', kbins=np.logspace(-2, 1, 30), deconv_cic=True)
pv = f.power('velocity', kbins=...)         # vector spectrum |v|² summed
ptheta = f.power('theta', ...)               # divergence ∇·v in k-space

# Cross-spectra:
pdv = f.cross_power('delta', 'velocity_par', ...)   # δ × v_∥(k)

# Real-space correlations (FFT-based):
xi = f.correlation('delta', rbins=...)
psi = f.correlation('velocity', rbins=...)

# Folding for sub-Nyquist extension (existing compute_pk feature):
pk_folded = f.power('delta', fold_levels=[4, 16], ...)
```

`ICField` holds the loaded HDF5 metadata (box size, redshift, $h$,
particle count) so any cached spectrum knows its physical units. The
result dict is what gets written to disk by `io.write_table`.

## Theory module sketch

```python
from icpipe import theory

th = theory.LinearTheory.from_class('data/class_pk_z200_pk.dat',
                                     z=200, h=0.6736, Omega_m=0.3158)

th.Pk(k)               # P_δ(k)  in (Mpc/h)³
th.Pv(k)               # P_v(k) = (aHf/k)² P_δ(k)  in (km/s)²·(Mpc/h)³
th.xi(r)               # ξ(r) via Hankel transform
th.psi(r)              # ψ(r) via Hankel transform with no k² weight
th.f_growth            # f(z) = dlnD/dlna ≈ Ω_m(z)^0.55
th.aHf                 # convenience: a·H·f at z (km/s/Mpc)
```

## CLI tools become thin wrappers

```python
# scripts/compute_pk.py:
from icpipe import ICField, io
def main():
    args = parse_args()
    f = ICField(args.hdf5, ngrid=args.ngrid, interlace=args.interlace)
    pk = f.power('delta', kbins=..., fold_levels=args.fold)
    io.write_pk(args.out, pk)
```

Same for `compute_pv.py`. `plot_ic.py` uses `icpipe.theory` for the
overlay curves.

## Install / import strategy

Two options:

- **(i)** `pip install -e .` from project root with a minimal
  `pyproject.toml`. Cleanest. Then `from icpipe import ICField` works
  anywhere in the conda env.
- **(ii)** Path manipulation at top of each script:
  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from icpipe import ICField
  ```
  Uglier but no install step.

Recommend **(i)**. One-time `pip install -e .` in the cosmo conda env.

## Testing

Add `icpipe/tests/` with:
- `test_cic.py`: CIC mass conservation, simple analytic cases.
- `test_window.py`: deconvolution recovers theoretical W²(k).
- `test_binning.py`: radial binning produces correct bin centers
  and counts.
- `test_theory.py`: theory module reproduces known limits
  ($P_v(k)/P_\delta(k) = (aHf/k)^2$).

Run via `pytest icpipe/tests/`.

## Implementation phases

1. **Phase 1: package skeleton.** Create directory, `__init__.py`,
   `pyproject.toml`, install in conda env.
2. **Phase 2: extract reusable bits from `compute_pk.py`.** Move CIC,
   window deconvolution, binning into `field.py`, `windows.py`,
   `binning.py`. Refactor `compute_pk.py` as a thin wrapper.
   Verify pipeline still produces identical output.
3. **Phase 3: implement `ICField.power('velocity')`.** Add the
   velocity-CIC + FFT path in `field.py`. Write `compute_pv.py` on
   top.
4. **Phase 4: theory module.** Pull linear-theory predictions out of
   `plot_ic.py` into `theory.py`. Refactor `plot_ic.py` to import.
5. **Phase 5: 4-panel demo figure.** New script (probably
   `scripts/plot_pk_pv_box_comparison.py`) that runs the package on
   ICs at multiple box sizes and assembles the 4-panel comparison.
6. **Phase 6: tests.** Add the test files and a `make test` target
   if useful.
7. **Phase 7: docs.** Update `CLAUDE.md`, `README.md` to point at
   the package and the new `compute_pv.py`.

Each phase ends with a working pipeline, so nothing breaks mid-way.

## Velocity-PK physics (summary, with references)

This package is what makes the velocity diagnostic feasible as a
first-class measurement, not a tacked-on script. The physics
motivation (preserved from `velocity_pk_plan.md`):

- Velocity field: $\mathbf v(\mathbf k) = (i\mathbf k/k^2)\,aHf\,\delta(\mathbf k)$
  → $P_v(k) = (aHf)^2\,P_\delta(k)/k^2$.
- Variance integrals: $\sigma_\delta^2 \propto \int k^2 P\,dk$ vs
  $\sigma_v^2 \propto \int P\,dk$. Velocity is large-scale-dominated;
  density is small-scale-dominated.
- Finite-box truncation (no modes below $k_{\rm fund}=2\pi/L$) hurts
  velocity statistics much more than density statistics.

References:
- **Klypin & Prada 2019** (MNRAS 489, 1684) — general finite-box
  effects on density statistics; effects negligible above $L\sim 1\,h^{-1}$Gpc.
- **Chuang+ 2026** (arXiv:2602.04485) — direct demonstration of the
  velocity-correlation suppression for small boxes; $L=1\,h^{-1}$Gpc
  is suppressed by 5–10% at $r\sim 100\,h^{-1}$Mpc; sub-volumes of a
  larger box keep the parent's suppression; introducing $k_{\min}$ as
  a free parameter recovers unbiased $f\sigma_8$.

This will go into the `cosmo_ic.tex` "starting redshifts and
convergence" section once the demo figure is in.

## Files this will create / touch

- `pyproject.toml` — new (minimal, for `pip install -e .`).
- `icpipe/` — new package directory.
- `scripts/compute_pk.py` — refactored into thin wrapper.
- `scripts/compute_pv.py` — new, uses `ICField`.
- `scripts/plot_ic.py` — uses `icpipe.theory`.
- `scripts/plot_pk_pv_box_comparison.py` — new (4-panel demo).
- `notes/cosmo_ic.tex` — new subsection on box-size sensitivity of
  velocity statistics, citing Klypin+ 2019 and Chuang+ 2026.
- `CLAUDE.md`, `README.md` — point at the package and `compute_pv.py`.
