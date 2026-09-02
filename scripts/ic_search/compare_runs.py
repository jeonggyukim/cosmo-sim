#!/usr/bin/env python3
"""Compare two sweeps on the seeds they have in common.

Used to check that a change to how the fields are generated -- batching several
seeds into one monofonIC call, say -- leaves the measurements alone. The spectra
are the quantities that must agree exactly: they depend only on the field and the
pencil geometry, not on any analysis choice that may differ between the runs.

Quantities computed from a smoothed field are compared only when both runs used
the same smoothing radii, since a run with different radii is measuring a
different thing and disagreement would mean nothing.

    python compare_runs.py --a DIR --b DIR

Reports, per dataset, whether every value on the common seeds is equal.
"""
import argparse, glob, os, sys
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="reference sweep directory")
ap.add_argument("--b", required=True, help="sweep directory to check")
ap.add_argument("--max-seeds", type=int, default=0,
                help="stop after this many common seeds (0 = all)")
A = ap.parse_args()

# Datasets whose values depend on the smoothing radius.
SMOOTHED = {"shear", "dbar", "lambda", "webtype", "contrast",
            "shear_box", "dbar_box", "lambda_box", "webtype_box",
            "shear_interior", "dbar_interior", "lambda_interior"}
# Datasets that are per-file metadata rather than per-seed measurements.
META = {"k", "P_theory", "P_win", "pencil_axis", "pencil_i", "pencil_j",
        "smooth_R", "margin_cells", "skipped"}


def index(path):
    """seed -> (file, row) for every seed in a sweep, plus its smoothing radii."""
    where, radii = {}, None
    for fn in sorted(glob.glob(f"{path}/chunk_*.hdf5")):
        with h5py.File(fn) as f:
            if "seed" not in f:
                continue
            if radii is None and "smooth_R" in f:
                radii = f["smooth_R"][:]
            for row, s in enumerate(f["seed"][:]):
                where[int(s)] = (fn, row)
    if not where:
        raise SystemExit(f"no chunk files with seeds under {path}")
    return where, radii


ia, ra = index(A.a)
ib, rb = index(A.b)
common = sorted(set(ia) & set(ib))
if A.max_seeds:
    common = common[:A.max_seeds]
if not common:
    raise SystemExit(f"no seeds in common: {A.a} has {len(ia)}, {A.b} has {len(ib)}")

same_radii = ra is not None and rb is not None and np.array_equal(ra, rb)
print(f"{A.a}\n  {len(ia):,} seeds, smoothing radii "
      f"{np.round(ra, 2) if ra is not None else 'none'}")
print(f"{A.b}\n  {len(ib):,} seeds, smoothing radii "
      f"{np.round(rb, 2) if rb is not None else 'none'}")
print(f"\ncomparing {len(common)} seeds in common: {common[0]}-{common[-1]}")
if not same_radii:
    print("radii differ, so smoothed quantities are skipped rather than compared")
print()

# Which per-seed datasets both runs carry.
with h5py.File(ia[common[0]][0]) as fa, h5py.File(ib[common[0]][0]) as fb:
    keys = sorted((set(fa) & set(fb)) - META - {"seed"})

verdict = {}
for k in keys:
    if k in SMOOTHED and not same_radii:
        verdict[k] = ("skipped", "different smoothing radii")
        continue
    worst, nbad = 0.0, 0
    for s in common:
        fna, ra_ = ia[s]
        fnb, rb_ = ib[s]
        with h5py.File(fna) as fa, h5py.File(fnb) as fb:
            x, y = fa[k][ra_], fb[k][rb_]
        if x.shape != y.shape:
            verdict[k] = ("shape", f"{x.shape} vs {y.shape}")
            nbad = -1
            break
        if not np.array_equal(x, y):
            nbad += 1
            d = np.abs(np.asarray(x, float) - np.asarray(y, float))
            worst = max(worst, float(d.max()))
    if nbad == 0:
        verdict[k] = ("identical", "")
    elif nbad > 0:
        verdict[k] = ("differs", f"{nbad}/{len(common)} seeds, max|d| {worst:.3e}")

width = max(len(k) for k in keys)
bad = 0
for k in keys:
    state, note = verdict[k]
    print(f"  {k:<{width}}  {state:<9} {note}")
    bad += state in ("differs", "shape")

print()
if bad == 0:
    print(f"PASS: every comparable dataset agrees on all {len(common)} common seeds")
else:
    print(f"FAIL: {bad} dataset(s) disagree")
sys.exit(1 if bad else 0)
