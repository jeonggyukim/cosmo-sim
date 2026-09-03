# Reorganising `scripts/ic_search/`

Deferred until the 20,000-seed sweep (job 496916, run tag `xi20k`) has finished.
`sweep.sbatch` does `cd "$WORKDIR/scripts"` and then runs `pencil_seed_sweep.py`
from there, and each array task reads that path when it starts rather than when
the array is submitted. Moving files and re-syncing while an array is in flight
breaks every task that has not launched yet.

## Why

36 files sit flat in one directory. The production sweep, the analysis, thirteen
plotting scripts, the tests and two watcher shell scripts are indistinguishable
by location, so which file is on the critical path for a cluster run is not
visible without reading them.

## Proposed layout

```
scripts/ic_search/
  README.md
  lib/        paths.py  chunkio.py
  sweep/      pencil_seed_sweep.py  sweep.sbatch  topup.sbatch
              collect_missing.py  merge_sweeps.py
  analysis/   run_analysis.py  analyze_environment.py  analyze_sweep.py
              fit_amplitude.py  check_xi_skew.py  check_selected_xi.py
  figures/    plot_*.py
  tests/      test_estimators.py  test_boundary.sh  test_batch_sweep.sh
              test_seed_loop.sh
  watch/      auto_analyze.sh  watch_job.sh
```

## What breaks, and has to be fixed in the same change

1. Every script reaches `paths` and `chunkio` by being in the same directory as
   them. They need `sys.path` pointed at `lib/`, or `lib` made a package.
2. `run_analysis.py` builds each step's command as `os.path.join(HERE, name)`.
3. `sweep.sbatch` changes into `scripts/` and runs the sweep by bare filename.
4. The rsync that puts scripts on the cluster filters `--include="*.py"` with no
   recursion, so it would copy nothing from the new subdirectories.
5. `README.md`, `RUNS.md` and the repository `CLAUDE.md` name files at the
   current paths. `CLAUDE.md` requires README and CLAUDE to be updated in the
   same commit as any structural change.

## Acceptance gate

The move is checkable now in a way it was not before, and all three should pass
unchanged afterwards:

- `test_estimators.py` — the estimators against closed-form answers on a
  constant field, a masked constant field and a plane wave.
- `test_boundary.sh` — the inside/outside tidal source split closing on the
  measured shear.
- `run_analysis.py` — all eleven analysis steps against an existing sweep
  directory.

Then re-sync to the cluster and verify by checksum, not by the absence of an
rsync error message: a silent sync failure in this session left the cluster
running week-old code through a full smoke test.
