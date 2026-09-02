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
# weaker would have missed the first version of this code, which drew the right
# realisation at the wrong amplitude.
#
# Both DoFixing settings are tested. Amplitude fixing is applied when the noise
# is normalised, which is the step the batch path has to repeat for every seed,
# so it is exactly where the two paths could diverge.
#
#   scripts/ic_search/test_seed_loop.sh [NGRID] [NSEEDS]
#
# Needs the lagrangian-density fork built and the reference config that
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
# The comparison needs h5py, which the system python does not have.
PYTHON=${PYTHON:-"conda run -n cosmo python"}

for f in "$BIN" "$TEMPLATE"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
echo "binary   $BIN"
echo "template $TEMPLATE"
echo "work     $WORK"
echo "grid     ${NGRID}^3, seeds ${SEED0}..$(( SEED0 + NSEEDS - 1 ))"

# CLASS faults on startup often enough that a test must tolerate it. The failure
# happens before the seed is used and a rerun clears it.
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
    local dir=$1 seed=$2 count=$3 fixing=$4
    mkdir -p "$dir"
    sed -e "s/^GridRes.*/GridRes         = $NGRID/" \
        -e "s/^seed.*/seed            = $seed/" \
        -e "s/^DoFixing.*/DoFixing        = $fixing/" \
        -e "s|^filename.*|filename        = $dir/deltaq.hdf5|" \
        "$TEMPLATE" > "$dir/c.conf"
    if [ "$count" -gt 1 ]; then
        sed -i'' -e "s/^\[setup\]/[setup]\nSeedCount = $count/" "$dir/c.conf"
    fi
}

cat > "$WORK/compare.py" <<'PY'
import sys, glob
import numpy as np, h5py

work, tag, seed0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
bad = 0
for i in range(n):
    seed = seed0 + i
    ref = f"{work}/{tag}_one_{seed}/deltaq.hdf5"
    hit = glob.glob(f"{work}/{tag}_batch/deltaq_seed{seed}.hdf5")
    if not hit:
        print(f"  seed {seed}: batch file missing"); bad += 1; continue
    with h5py.File(ref) as a, h5py.File(hit[0]) as b:
        for k in (x for x in a if x != "Header"):
            x, y = a[k][:], b[k][:]
            if np.array_equal(x, y):
                print(f"  seed {seed}  {k:16s} bit-identical")
            else:
                bad += 1
                r = np.corrcoef(x.ravel(), y.ravel())[0, 1]
                print(f"  seed {seed}  {k:16s} DIFFERS  max|d| {np.abs(x-y).max():.3e}"
                      f"  corr {r:+.4f}  rms ratio {y.std()/x.std():.4f}")
sys.exit(1 if bad else 0)
PY

status=0
for fixing in no yes; do
    echo
    echo "=== DoFixing = $fixing ==="

    t0=$SECONDS
    for i in $(seq 0 $(( NSEEDS - 1 ))); do
        seed=$(( SEED0 + i ))
        write_conf "$WORK/${fixing}_one_$seed" "$seed" 1 "$fixing"
        run_monofonic "$WORK/${fixing}_one_$seed" || { status=1; continue; }
    done
    t_one=$(( SECONDS - t0 ))

    t0=$SECONDS
    write_conf "$WORK/${fixing}_batch" "$SEED0" "$NSEEDS" "$fixing"
    run_monofonic "$WORK/${fixing}_batch" || { status=1; continue; }
    t_batch=$(( SECONDS - t0 ))

    echo "one process per seed  : ${t_one}s"
    echo "one process, all seeds: ${t_batch}s"
    [ "$t_batch" -gt 0 ] && echo "speedup: $(awk -v a=$t_one -v b=$t_batch 'BEGIN{printf "%.2fx", a/b}')"

    $PYTHON "$WORK/compare.py" "$WORK" "$fixing" "$SEED0" "$NSEEDS" || status=1
done

echo
if [ "$status" -eq 0 ]; then echo "PASS"; else echo "FAIL"; fi
exit $status
