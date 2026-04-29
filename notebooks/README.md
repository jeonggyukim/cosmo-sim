# icpipe example notebooks

Worked examples of the `icpipe` package on real IC data from this project.

## Prerequisites

```bash
# from the project root
conda activate cosmo                  # or your env
pip install -e ".[plot]"              # icpipe + matplotlib
./run_pipeline.sh                     # generate at least one IC + CLASS P(k)
```

The notebooks read paths under `data/` (relative to the project root).
A small chdir cell at the top of each notebook auto-corrects if you
launched Jupyter from this `notebooks/` directory.

## Files

| Notebook | What it shows |
|---|---|
| `01_quickstart.ipynb` | Load one IC HDF5, compute `P_δ(k)` and `P_v(k)` via `ICField`, overlay against `LinearTheory` from CLASS. The "ratio to theory" plot at the end previews the box-size effect. |
| `02_box_size_sensitivity.ipynb` | Same observables across multiple box sizes. The velocity ratio drops below 1 (more for smaller boxes); the density ratio stays at 1. Ends with the analytic missing-variance integral that explains the suppression. References Klypin & Prada 2019 and Chuang+ 2026. |

## Reproducing the data

The default file names in the notebooks (`pk_n256_z200_L*.txt`,
`pv_n256_z200_L*.txt`) are produced by:

```bash
for L in 256 512 1024; do
  ./run_pipeline.sh --ngrid 256 --lbox $L --zstart 200
  python scripts/compute_pv.py data/ics_swift_n256_z200_L${L}.hdf5
done
```

The pipeline already runs `compute_pk.py` and `compute_xi_cic.py`;
`compute_pv.py` (new in this branch) is the velocity counterpart.
