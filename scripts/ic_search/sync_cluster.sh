#!/bin/bash
# Pull the finished chunks of a cluster sweep and rebuild the figures locally.
#
# Only the reduced chunk files travel; the delta(q) fields never leave the
# cluster. Chunks are immutable once written, so rsync copies each one exactly
# once and a repeated call costs almost nothing.
#
#   scripts/ic_search/sync_cluster.sh [RUNTAG]
#
# Figures are stamped with the time and the number of realizations they rest on,
# so a figure from a partial sweep can never be mistaken for the final one.

set -eo pipefail

RUNTAG=${1:-big128}
HOST=${CLUSTER_HOST:-g}
REMOTE=${CLUSTER_DIR:-/gpfs/jeonggyukim/monofonic-tests}
LOCAL=${MONOFONIC_TESTS:-$HOME/Documents/monofonic-tests}
HERE=$(cd "$(dirname "$0")" && pwd)

DEST=$LOCAL/data/$RUNTAG
mkdir -p "$DEST"
rsync -q "$HOST:$REMOTE/data/$RUNTAG/theory.hdf5" "$DEST/" 2>/dev/null || true
rsync -q --ignore-existing "$HOST:$REMOTE/data/$RUNTAG/chunk_*.hdf5" "$DEST/" 2>/dev/null || true

# `ls` with no match fails, and pipefail would end the script before the
# message below ever prints.
NCHUNK=$(find "$DEST" -maxdepth 1 -name "chunk_*.hdf5" | wc -l | tr -d ' ')
if [ "$NCHUNK" -eq 0 ]; then
    echo "no chunks yet for $RUNTAG; the first tasks have not finished"
    exit 0
fi

STAMP=$(date +%Y%m%d_%H%M)
OUT=$LOCAL/${RUNTAG}_${STAMP}
mkdir -p "$OUT"

conda run -n cosmo python "$HERE/analyze_environment.py" --data "$DEST" \
    | tee "$OUT/environment.txt"
conda run -n cosmo python "$HERE/analyze_environment.py" --data "$DEST" --keep 0.01 \
    | tee "$OUT/environment_keep1pct.txt"

echo "$NCHUNK chunks -> $OUT"
