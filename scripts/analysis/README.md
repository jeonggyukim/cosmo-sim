# `scripts/analysis/` — ad-hoc IC inspection tools

CLIs that are **not** part of `run-pipeline`.  Each one is run by
hand to inspect or compare ICs already produced by the pipeline.

For the pipeline-step CLIs (`compute-pk`, `compute-pv`, ...) see
`icpipe/README.md`.

## Tools

### `check_ic.py`

Verifies that the DC (k=0) modes of the displacement and velocity
fields in a MUSIC2-generated IC are zero.  Reports two independent
diagnostics — the Lagrangian one from displacement = particle − lattice,
and the Eulerian one from the CIC-grid mean velocity.  Also prints the
mass resolution and the recommended SWIFT force-softening range
(ε = Δx/40 … Δx/25).

```bash
conda run -n cosmo python scripts/analysis/check_ic.py data/ics_swift_n256_z200_L1024.hdf5
conda run -n cosmo python scripts/analysis/check_ic.py --explain  data/ics_swift_*.hdf5
conda run -n cosmo python scripts/analysis/check_ic.py --hist hist.png data/ics_swift_n256_z200_L1024.hdf5
```

A correctly generated IC has both DC residuals well below 10⁻⁵ × RMS.

### `plot_box_size_comparison.py`

4-panel figure demonstrating that the velocity field is much more
sensitive to long-wavelength modes than the density field: smaller
boxes truncate modes below k_fund = 2π/L and so suppress velocity
statistics far more than density ones.  Overlays measurements from
several IC box sizes against a single linear-theory prediction.

```bash
conda run -n cosmo python scripts/analysis/plot_box_size_comparison.py
# expects: data/pk_n256_z200_L{256,512,1024}.txt,
#          data/pv_n256_z200_L{256,512,1024}.txt,
#          data/xi_cic_n256_z200_L{256,512,1024}.txt,
#          data/vel_cic_n256_z200_L{256,512,1024}.txt,
#          data/class_pk_z200_pk.dat
# output:  plots/box_size_comparison.png
```

### `plot_dr_histogram.py`

Histogram of dr/dx — the per-particle displacement (from the nearest
unperturbed lattice point) in units of the grid spacing.  Useful for
comparing several ICs side by side (different z_start, different
2LPT/3LPT, different padding).

```bash
conda run -n cosmo python scripts/analysis/plot_dr_histogram.py \
    data/ics_swift_n256_z200_L1024.hdf5 data/ics_swift_n256_z400_L1024.hdf5
```

### `xi_sweep.py`

Measures ξ(r) and P(k) on a pencil-beam sub-volume cut from a synthetic
Gaussian field, over many random seeds, then overlays the realisations.
Two subcommands so the expensive half is not repeated every time the
plot is restyled: `compute` writes one `.npz` per seed and skips seeds
already on disk, so a sweep is resumable and can be widened later;
`plot` loads every `.npz` in a directory and overlays them with a median
and a 16–84 percentile band.

Only the padded-FFT estimator runs during a sweep. `--check-identity`
additionally runs the direct lag sum and records the worst per-bin
relative difference, which is worth doing on one or two seeds as a spot
check and wasteful on all of them — the two are the same estimator, as
derived in `notes/xi_estimators.pdf`. At the default 128³ grid a seed
costs about 0.1 s with the FFT alone and about 0.5 s with the direct sum
as well.

The field generator, both estimators, and the P(k) binning are imported
from `notes/figures/xi_estimators/plot_xi_identity.py`, so the geometry
(box size, pencil fraction, per-axis periodicity) is defined in exactly
one place.

```bash
# 24 seeds, resumable
conda run -n cosmo python scripts/analysis/xi_sweep.py compute \
    --seeds 1000 1023 --outdir data/xi_sweep

# spot-check the estimator identity on one seed
conda run -n cosmo python scripts/analysis/xi_sweep.py compute \
    --seeds 1000 1000 --outdir data/xi_sweep --check-identity --force

# overlay everything found
conda run -n cosmo python scripts/analysis/xi_sweep.py plot \
    --outdir data/xi_sweep --output plots/xi_sweep.png
```

`--also-txt` additionally writes the `xi_*.txt` column format that the
HR-ICs-pipeline plotting scripts read.
