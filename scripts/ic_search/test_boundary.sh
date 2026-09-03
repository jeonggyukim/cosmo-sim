#!/bin/bash
# Check the two boundary-aware measurements against identities they must satisfy.
#
# --source-split decomposes the region's tidal field by where its source sits,
# T[delta] = T[M delta] + T[(1-M) delta]. Poisson is linear, so the three terms
# it records -- inside, outside, and the cross term -- must sum to the shear the
# sweep already measures, to round-off. That is a strong check: it fails if the
# mask, the smoothing kernel or the tensor assembly disagree between the two
# paths by so much as a DC mode.
#
# --kernel-weight reweights each cell by the fraction of its smoothing kernel
# that fell inside the region. Two things must hold: the weight is a fraction,
# so it stays in (0, 1], and as the radius falls well below the region width the
# weighted average must approach the unweighted one, since almost every kernel
# then lies inside.
#
#   scripts/ic_search/test_boundary.sh [NGRID] [NSEEDS]

set -uo pipefail

NGRID=${1:-64}
NSEEDS=${2:-2}
SEED0=${SEED0:-5101}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$(mktemp -d)}
PYTHON=${PYTHON:-"conda run -n cosmo python"}
THREADS=${THREADS:-4}
mkdir -p "$WORK"

echo "grid ${NGRID}^3, seeds ${SEED0}..$(( SEED0 + NSEEDS - 1 ))"
echo "work $WORK"
echo

$PYTHON "$HERE/pencil_seed_sweep.py" \
    --seed0 "$SEED0" --nseeds "$NSEEDS" --ngrid "$NGRID" \
    --species matter --npencils 6 --environment \
    --smooth-frac 0.15 0.5 1.0 --interior-margin 1.0 \
    --kernel-weight --source-split 3 \
    --nthreads "$THREADS" --compact --out "$WORK/run" > "$WORK/run.log" 2>&1
rc=$?
echo "sweep: exit $rc"
if [ $rc -ne 0 ]; then tail -20 "$WORK/run.log"; exit 1; fi
grep -E "kernel fraction|R = |check " "$WORK/run.log"
echo

$PYTHON - "$WORK/run" <<'PY'
import glob, sys
import numpy as np, h5py

fn = sorted(glob.glob(f"{sys.argv[1]}/chunk_*.hdf5"))[0]
bad = 0
with h5py.File(fn) as f:
    R = f["smooth_R"][:]
    src = f["shear_src"][:]          # (nseed, nR, nsplit, 3) in, out, cross
    ip = f["split_pencils"][:]
    shear = f["shear"][:]            # (nseed, nR, npencil)
    dsrc = f["dbar_src"][:]          # (nseed, nR, nsplit, 2)
    dbar = f["dbar"][:]
    kw = f["shear_kw"][:]
    dkw = f["dbar_kw"][:]

    print("source split closes  (sum of the three terms / measured <s^2>)")
    for r, Rv in enumerate(R):
        got = src[:, r].sum(-1)
        want = shear[:, r][:, ip]**2
        ratio = got/want
        print(f"  R = {Rv:6.2f} Mpc/h   {ratio.min():.9f} .. {ratio.max():.9f}")
        if not np.allclose(ratio, 1.0, rtol=1e-8):
            bad += 1; print("    FAIL: does not close")

    print("\nmean overdensity splits  (inside + outside / measured)")
    for r, Rv in enumerate(R):
        got = dsrc[:, r].sum(-1)
        want = dbar[:, r][:, ip]
        d = np.abs(got - want)/np.abs(want).max()
        print(f"  R = {Rv:6.2f} Mpc/h   max residual {d.max():.2e}")
        if d.max() > 1e-8:
            bad += 1; print("    FAIL: does not close")

    print("\nfraction of the region's shear variance sourced inside it")
    for r, Rv in enumerate(R):
        frac = src[:, r, :, 0].mean()/src[:, r].sum(-1).mean()
        cross = src[:, r, :, 2].mean()/src[:, r].sum(-1).mean()
        print(f"  R = {Rv:6.2f} Mpc/h   inside {frac:6.3f}   cross {cross:+6.3f}")

    print("\nkernel weighting against the plain average")
    for r, Rv in enumerate(R):
        ratio = kw[:, r]/shear[:, r]
        print(f"  R = {Rv:6.2f} Mpc/h   shear ratio {ratio.mean():.4f}"
              f"   dbar shift {np.abs(dkw[:, r] - dbar[:, r]).max():.3e}")
        if not np.all(np.isfinite(kw[:, r])):
            bad += 1; print("    FAIL: not finite")

print("\nFAILURES:", bad)
sys.exit(1 if bad else 0)
PY
rc=$?
echo
[ $rc -eq 0 ] && echo "all checks passed" || echo "CHECKS FAILED"
exit $rc
