# Sweep runs

One entry per production sweep, with everything needed to regenerate it. Seeds
are derived from the array task index, so the seed of every realization is fixed
by the entry below and not by anything recorded at run time. Each chunk file also
stores the `seed` list it actually measured and a `skipped` list of seeds
monofonIC could not generate, so the data is self-describing even without this
file.

## big128 — 100,000 realizations at N = 128

| | |
|---|---|
| Slurm job | 490017 on grammar, array 0-499 |
| Seeds | 100000-199999; task *t* runs 100000 + 200*t* … + 199 |
| Grid, box | 128^3, 700 Mpc/h |
| Species | matter |
| Pencils | 24 per realization, of the 192 (1/8 of the box in two axes, full length in the third) |
| Environment | tidal shear, mean overdensity, eigenvalues, web type, bulk flow, neighbour contrast, at R = 20 and 40 Mpc/h |
| Amplitude fixing | DoFixing = no |
| Output | `/gpfs/jeonggyukim/monofonic-tests/data/big128/chunk_<seed0>_<n>.hdf5` |
| Started | 2026-09-03 |

Submitted with:

```bash
sbatch --array=0-499 --export=ALL,NPER=200,SEEDBASE=100000,NGRID=128,RUNTAG=big128 \
       sweep.sbatch
```

The array ran unthrottled after `scontrol update jobid=490017 arraytaskthrottle=0`;
the throttle changes only how many tasks run at once, never which seeds they use.

Code:

- monofonIC fork `github.com/jeonggyukim/monofonIC`, branch `lagrangian-density`,
  commit `291fd12`, built with `-DENABLE_PLT=ON -DENABLE_MPI=ON`.
- cosmo-sim commit `f226065` for `pencil_seed_sweep.py` and `sweep.sbatch`.
- Modules: gnu12/12.2.0, openmpi4/4.1.5, fftw/3.3.10, hdf5/1.14.0, gsl/2.7.1.

Cosmology, from `ref/deltaq_n64_L700.conf`: Omega_m = 0.3, Omega_b = 0.049,
sigma_8 = 0.8, n_s = 0.9624, z_start = 200, 2LPT, `transfer = CLASS`,
`TransferComponent = matter` (Michaux+2020 back-scaling).

Three values in that template do **not** describe the run: `GridRes`, `seed` and
`DoFixing` are rewritten by `pencil_seed_sweep.py` for every realization, to the
grid given by `--ngrid`, the seed being swept, and `--dofixing` (default `no`).
The template's `GridRes = 64`, `seed = 12345` and `DoFixing = yes` are never used
by a sweep.

Measured cost: 67 s per seed on one grammar core. A 200-seed task takes 3.7 h,
and the array totals 100,000 x 67 s = 1,860 core-hours.

## big128b — a second 100,000 at N = 128, with the added measurements

| | |
|---|---|
| Slurm job | 491168 on grammar, array 0-499, `--dependency=afterany:490017` |
| Seeds | 200000-299999; task *t* runs 200000 + 200*t* … + 199 |
| Smoothing | radii at 0.1, 0.25 and 0.5 of the pencil width: 8.75, 21.9, 43.75 Mpc/h |
| Interior margin | one smoothing radius trimmed from each long face |
| Also recorded | whole-box shear, eigenvalues and web fractions; interior variants |
| Chunk files | written every 25 seeds, so the run can be analysed while it goes |
| Output | `/gpfs/jeonggyukim/monofonic-tests/data/big128b/` |

Submitted with:

```bash
sbatch --dependency=afterany:490017 --array=0-499 \
       --export=ALL,NPER=200,SEEDBASE=200000,NGRID=128,RUNTAG=big128b,FLUSH=25 \
       sweep.sbatch
```

Everything else matches big128, so the two can be merged for the quantities they
share. They do not share the smoothing radii: big128 used 20 and 40 Mpc/h fixed,
this run uses fractions of the region width, and only the 0.25 fraction
(21.9 Mpc/h) is close to one of them.

## Earlier runs

| tag | seeds | N | notes |
|---|---|---|---|
| `web_n128_p1..p4` (local) | 6000-6399 | 128 | 374 usable; p1 and p4 aborted on a seed monofonIC could not generate, before the skip fix |
| `env_n128_L700_x400` (local) | 5000-5399 | 128 | shear and mean overdensity only, no eigenvalues |
| `pencil_sweep_n64_L700*` (local) | 1000-1199 | 64 | first sweeps, all three species |
| `smoketest` (grammar) | 900000-900019 | 64 | array plumbing check |
| `timing128` (grammar) | 950000-950002 | 128 | per-seed cost measurement |

Local and cluster seed ranges do not overlap, so any of these can be merged with
`merge_sweeps.py`, which refuses duplicate seeds.
