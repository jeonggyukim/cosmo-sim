#!/usr/bin/env python3
"""List the seeds a sweep was meant to measure but did not, so they can be rerun.

Seeds go missing two ways. A seed is *skipped* when monofonIC will not produce a
field for it: CLASS faults on startup, before the seed has been used for
anything, and the sweep records the seed and moves on. A seed is *absent* when
the task that owned it never wrote its chunk at all, because it failed or ran
past its time limit.

This finds both, by comparing the seeds actually present in the chunk files
against the range the run was supposed to cover. It writes one seed per line, in
the form pencil_seed_sweep.py --seed-list reads.

    python collect_missing.py --data DIR --expect-start 100000 --expect-count 100000
    # then, on the cluster:
    sbatch --array=0-19 --export=ALL,SEEDFILE=missing_seeds.txt,NPER=100 topup.sbatch

Neither kind of loss selects on the realization, so rerunning them is a matter
of completeness rather than of correctness: the transfer function CLASS crashes
in is identical for every seed, and a task dying is unrelated to the fields it
was carrying.
"""
import argparse, glob, os
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="sweep directory holding chunk_*.hdf5")
ap.add_argument("--expect-start", type=int, default=None,
                help="first seed the run was supposed to cover")
ap.add_argument("--expect-count", type=int, default=None,
                help="how many consecutive seeds the run was supposed to cover")
ap.add_argument("--out", default=None, help="output file (default missing_seeds.txt in --data)")
A = ap.parse_args()

chunks = sorted(glob.glob(f"{A.data}/chunk_*.hdf5"))
if not chunks:
    raise SystemExit(f"no chunk_*.hdf5 under {A.data}")

measured, skipped = [], []
for fn in chunks:
    with h5py.File(fn) as f:
        if "seed" in f:
            measured.append(f["seed"][:])
        if "skipped" in f:
            skipped.append(f["skipped"][:])
measured = np.concatenate(measured) if measured else np.array([], dtype=np.int64)
skipped = np.concatenate(skipped) if skipped else np.array([], dtype=np.int64)

print(f"{len(chunks)} chunks, {len(measured)} seeds measured, "
      f"{len(skipped)} recorded as skipped")

missing = np.unique(skipped)
if A.expect_start is not None and A.expect_count is not None:
    expected = np.arange(A.expect_start, A.expect_start + A.expect_count)
    absent = np.setdiff1d(expected, measured)
    print(f"expected {len(expected)} seeds over "
          f"{A.expect_start}-{A.expect_start + A.expect_count - 1}; "
          f"{len(absent)} are not present")
    print(f"   of those, {len(np.intersect1d(absent, skipped))} recorded as skipped, "
          f"{len(np.setdiff1d(absent, skipped))} in chunks that were never written")
    missing = absent

out = A.out or os.path.join(A.data, "missing_seeds.txt")
np.savetxt(out, missing, fmt="%d")
frac = 100*len(missing)/max(1, len(missing) + len(measured))
print(f"wrote {len(missing)} seeds to {out}  ({frac:.2f}% of the intended total)")
