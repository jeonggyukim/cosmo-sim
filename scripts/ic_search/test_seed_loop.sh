#!/bin/bash
# Check that generating many seeds in one monofonIC process gives the same fields
# as generating them one process at a time, and measure what it saves.
#
# The transfer function, the growth factors and the cosmology do not depend on
# the seed, so a run that writes N seeds needs one CLASS evaluation rather than
# N. [setup] SeedCount turns that on, and this script is its acceptance test.
#
# The test is exact rather than statistical: a seed generated inside a batch must
# produce a delta(q) bit-identical to the same seed generated alone. Anything
# else means the loop has disturbed the random number generator.
#
#   scripts/ic_search/test_seed_loop.sh [NGRID] [NSEEDS]
#
# Needs the lagrangian-density fork built, and the reference config that
# scripts/ic_search/paths.py points at.

set -uo pipefail

NGRID=${1:-64}
NSEEDS=${2:-3}
SEED0=${SEED0:-1001}
BIN=${MONOFONIC_BIN:-$HOME/Library/CloudStorage/Dropbox/Projects/monofonIC-lagrangian-density/build/monofonIC}
TESTS=${MONOFONIC_TESTS:-$HOME/Documents/monofonic-tests}
TEMPLATE=${MONOFONIC_REF_CONF:-$TESTS/n64_deltaq_z200_L700/deltaq_n64_L700.conf}
WORK=${WORK:-$(mktemp -d)}
THREADS=${OMP_NUM_THREADS:-8}

for f in "$BIN" "$TEMPLATE"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
echo "binary   $BIN"
echo "template $TEMPLATE"
echo "work     $WORK"
echo "grid     ${NGRID}^3, seeds ${SEED0}..$(( SEED0 + NSEEDS - 1 ))"
echo

# CLASS faults on startup often enough that a test must tolerate it; the failure
# is unrelated to the seed and a rerun clears it.
run_monofonic() {
    local dir=$1 attempt
    for attempt in 1 2 3 4 5 6; do
        if ( cd "$dir" && OMP_NUM_THREADS=$THREADS "$BIN" c.conf > run.log 2>&1 ); then
            return 0
        fi
        sleep 2
    done
    echo "  monofonIC failed six times in $dir"
    return 1
}

write_conf() {
    local dir=$1 seed=$2 count=$3
    mkdir -p "$dir"
    sed -e "s/^GridRes.*/GridRes         = $NGRID/" \
        -e "s/^seed.*/seed            = $seed/" \
        -e "s/^DoFixing.*/DoFixing        = no/" \
        -e "s|^filename.*|filename        = $dir/deltaq.hdf5|" \
        "$TEMPLATE" > "$dir/c.conf"
    if [ "$count" -gt 1 ]; then
        sed -i'' -e "s/^\[setup\]/[setup]\nSeedCount = $count/" "$dir/c.conf"
    fi
}

# One process per seed, the current behaviour, used as the reference.
t0=$SECONDS
for i in $(seq 0 $(( NSEEDS - 1 ))); do
    seed=$(( SEED0 + i ))
    write_conf "$WORK/one_$seed" "$seed" 1
    run_monofonic "$WORK/one_$seed" || exit 1
done
t_one=$(( SECONDS - t0 ))
echo "one process per seed : ${t_one}s for $NSEEDS seeds"

# All seeds from one process.
t0=$SECONDS
write_conf "$WORK/batch" "$SEED0" "$NSEEDS"
run_monofonic "$WORK/batch" || exit 1
t_batch=$(( SECONDS - t0 ))
echo "one process, all seeds: ${t_batch}s for $NSEEDS seeds"
if [ "$t_batch" -gt 0 ]; then
    echo "speedup: $(python3 -c "print(f'{$t_one/$t_batch:.2f}x')")"
fi
echo

python3 - "$WORK" "$SEED0" "$NSEEDS" <<'PY'
import sys, glob, h5py, numpy as np
work, seed0, nseeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bad = 0
for i in range(nseeds):
    seed = seed0 + i
    ref = f"{work}/one_{seed}/deltaq.hdf5"
    bat = sorted(glob.glob(f"{work}/batch/deltaq_seed{seed}.hdf5"))
    if not bat:
        print(f"seed {seed}: batch file missing"); bad += 1; continue
    with h5py.File(ref) as a, h5py.File(bat[0]) as b:
        keys = [k for k in a if k != "Header"]
        for k in keys:
            x, y = a[k][:], b[k][:]
            same = np.array_equal(x, y)
            print(f"seed {seed}  {k:16s} bit-identical: {same}"
                  f"{'' if same else f'   max|diff| {np.abs(x-y).max():.3e}'}")
            bad += (not same)
print("\nPASS" if bad == 0 else f"\nFAIL ({bad} mismatches)")
sys.exit(1 if bad else 0)
PY
