#!/bin/bash
# Check that batching the IC generation changes nothing the sweep measures.
#
# --batch-seeds B asks monofonIC for B consecutive seeds per call instead of one,
# which costs one CLASS evaluation instead of B. The fields themselves are
# already known to be bit-identical either way (test_seed_loop.sh); this checks
# the sweep on top of them, so that a mistake in how the batch is consumed --
# a field read for the wrong seed, a stale file left behind, an off-by-one in
# the run of consecutive seeds -- cannot pass unnoticed.
#
# Every array in the chunk file must match exactly.
#
#   scripts/ic_search/test_batch_sweep.sh [NGRID] [NSEEDS] [BATCH]

set -uo pipefail

NGRID=${1:-64}
NSEEDS=${2:-6}
BATCH=${3:-3}
SEED0=${SEED0:-3001}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$(mktemp -d)}
PYTHON=${PYTHON:-"conda run -n cosmo python"}
THREADS=${THREADS:-4}

echo "grid ${NGRID}^3, seeds ${SEED0}..$(( SEED0 + NSEEDS - 1 )), batch $BATCH"
echo "work $WORK"
echo

for mode in one batch; do
    b=1
    [ "$mode" = batch ] && b=$BATCH
    t0=$SECONDS
    $PYTHON "$HERE/pencil_seed_sweep.py" \
        --seed0 "$SEED0" --nseeds "$NSEEDS" --ngrid "$NGRID" \
        --species matter --npencils 6 --environment --smooth 20 40 \
        --nthreads "$THREADS" --batch-seeds "$b" --compact \
        --out "$WORK/$mode" > "$WORK/$mode.log" 2>&1
    rc=$?
    echo "$mode (--batch-seeds $b): exit $rc, $(( SECONDS - t0 ))s"
    if [ $rc -ne 0 ]; then tail -5 "$WORK/$mode.log"; exit 1; fi
done
echo

cat > "$WORK/compare.py" <<'PY'
import sys, glob
import numpy as np, h5py

work = sys.argv[1]
a_files = sorted(glob.glob(f"{work}/one/chunk_*.hdf5"))
b_files = sorted(glob.glob(f"{work}/batch/chunk_*.hdf5"))
if not a_files or not b_files:
    print("missing chunk files"); sys.exit(1)

bad = 0
with h5py.File(a_files[0]) as A, h5py.File(b_files[0]) as B:
    keys = sorted(set(A) | set(B))
    for k in keys:
        if k not in A or k not in B:
            print(f"  {k:18s} present in only one run"); bad += 1; continue
        x, y = A[k][:], B[k][:]
        if x.shape != y.shape:
            print(f"  {k:18s} shape {x.shape} vs {y.shape}"); bad += 1; continue
        if np.array_equal(x, y):
            print(f"  {k:18s} identical  {x.shape}")
        else:
            bad += 1
            d = np.abs(np.asarray(x, float) - np.asarray(y, float))
            print(f"  {k:18s} DIFFERS    max|d| {d.max():.3e}")
print()
print("PASS" if bad == 0 else f"FAIL ({bad} arrays differ)")
sys.exit(1 if bad else 0)
PY

$PYTHON "$WORK/compare.py" "$WORK"
