# `icpipe` — IC analysis library + pipeline CLIs

Python package backing the `cosmo-pipeline` repo.  Two roles:

1. **Library** (`icpipe.field`, `icpipe.theory`, `icpipe.io`, ...): import
   from notebooks or other scripts to load SWIFT ICs, build CIC grids,
   compute power spectra, and overlay linear-theory predictions.
2. **Pipeline CLIs** (`icpipe.cli.*`): each module exposes a `main()`
   wired into a `console_scripts` entry point in `pyproject.toml`, so
   the pipeline-step commands appear on `$PATH` after install.

## Install

From the repo root:

```bash
pip install -e .                # core: numpy / scipy / h5py
pip install -e ".[plot,test]"   # also matplotlib + pytest
pytest icpipe/tests/            # 17 tests
```

Inside the `cosmo` conda env:

```bash
conda activate cosmo
pip install -e ".[plot,test]"
```

## Console scripts (pipeline step commands)

After install, the following commands appear on `$PATH`:

| command           | module                       | one-liner |
|-------------------|------------------------------|-----------|
| `make-music-conf` | `icpipe.cli.make_music_conf` | expand the MUSIC2 config template for a given (N, z, L) |
| `make-rbins`      | `icpipe.cli.make_rbins`      | write a Corrfunc bin file (rmin = 2 × mean spacing, rmax = L/3) |
| `compute-pk`      | `icpipe.cli.compute_pk`      | estimate P(k) from IC particles (CIC + FFT) → `pk_*.txt` |
| `compute-pv`      | `icpipe.cli.compute_pv`      | estimate the velocity power spectrum P_v(k) → `pv_*.txt` |
| `plot-ic`         | `icpipe.cli.plot_ic`         | overlay measured + CLASS theory on `pk_*`, `xi_*`, `vel_cic_*` |

Each command also works as `python -m icpipe.cli.<name> ...`.
`run-pipeline` invokes these commands directly.

## Library API

### `ICField`

```python
from icpipe import ICField

f = ICField("data/ics_swift_n256_z200_L1024.hdf5",
            ngrid=256, interlace=True, h=0.6711,
            load_velocities=True)
res = f.power("delta")          # PowerSpectrumResult
res_v = f.power("velocity")
print(res.k, res.P, res.P_shot)
```

Loads SWIFT IC HDF5, caches CIC density / momentum / velocity Fourier
grids, and exposes `.power(field, ...)` for δ or velocity spectra
with optional interlacing, CIC-window deconvolution, and shot
subtraction.

### `LinearTheory`

```python
from icpipe import LinearTheory

th = LinearTheory.from_class("data/class_pk_z0_pk.dat",
                              z=200, h=0.6711, Omega_m=0.3155)
Pk, Pv = th.Pk, th.Pv             # callables
xi_arr  = th.xi(r_array)
psi_arr = th.psi(r_array)
```

Returns matched-units `Pk`, `Pv`, `xi`, `psi` from a single CLASS
table.  Internally back-scales with the matter-only growth factor
D₊ⁿᵒ⁻ʳᵃᵈ(z) to stay consistent with MUSIC2's ZeroRadiation=true output.

### `icpipe.io`

ASCII writers / readers for the table formats `compute_pk` and
`compute_pv` produce:

```python
from icpipe.io import read_pk, read_pv, write_pk, write_pv
data = read_pk("data/pk_n256_z200_L1024.txt")
```

Raw inspection readers (not used by `ICField` itself):

```python
from icpipe.io import read_wnoise, read_swift_ics, print_swift_ics_summary
w = read_wnoise("data/wnoise_0008.bin")            # (nx, ny, nz) ndarray
header, parts = read_swift_ics("data/ics_swift_n256_z200_L1024.hdf5")
print_swift_ics_summary("data/ics_swift_n256_z200_L1024.hdf5")
```

### `deposit`

```python
from icpipe import deposit
rho = deposit(positions, boxsize, ngrid)            # mass (CIC, order 2)
mom = deposit(positions, boxsize, ngrid, order=4,   # PCS assignment
              weights=velocities[:, 0])             # x-momentum
```

Periodic mass / scalar-weight deposit on an `ngrid³` grid using a B-spline
of the given interpolation order (NGP=1, CIC=2, TSC=3, PCS=4; Sefusatti
et al. 2016); the primitive `ICField` uses for every density / momentum /
velocity grid it builds.

## Tests + examples

- `pytest icpipe/tests/` — 17 unit tests (binning, field, theory, windows).
- `notebooks/01_quickstart.ipynb` — minimal end-to-end use.
- `notebooks/02_box_size_sensitivity.ipynb` — comparing P(k) across L.

## Layout

```
icpipe/
├── __init__.py         # public re-exports (ICField, LinearTheory, deposit, io, ...)
├── field.py            # ICField + deposit
├── theory.py           # LinearTheory.from_class(...)
├── io.py               # read/write_pk, read/write_pv, read_wnoise, read_swift_ics
├── binning.py          # radial binning helpers
├── windows.py          # CIC window function
├── cli/                # pipeline-step CLI modules (console_scripts)
└── tests/              # pytest suite
```
