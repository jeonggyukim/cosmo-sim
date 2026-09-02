#!/bin/bash
# Pull new chunks from a running sweep and re-run the analysis as they accumulate.
#
# Chunks are immutable once written, so rsync copies each exactly once and a
# repeated call is cheap. The analysis is re-run whenever the number of chunks
# has grown by STEP since the last time, and once more after the job leaves the
# queue. Each completed analysis prints one line naming its output directory, so
# a watcher can turn that into a notification.
#
#   auto_analyze.sh JOBID RUNTAG [STEP]
#
# STEP is in chunks, default 400. At 25 seeds per chunk that is 10,000
# realizations between looks, which is enough for the numbers to move.

set -uo pipefail

JOBID=${1:?usage: auto_analyze.sh JOBID RUNTAG [STEP]}
RUNTAG=${2:?usage: auto_analyze.sh JOBID RUNTAG [STEP]}
STEP=${3:-400}
INTERVAL=${INTERVAL:-60}
HOST=${CLUSTER_HOST:-g}
REMOTE=${CLUSTER_DIR:-/gpfs/jeonggyukim/monofonic-tests}
LOCAL=${MONOFONIC_TESTS:-$HOME/Documents/monofonic-tests}
KEEP=${KEEP:-0.01}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-"conda run -n cosmo python"}

DEST=$LOCAL/data/$RUNTAG
mkdir -p "$DEST"

count_local() { find "$DEST" -maxdepth 1 -name 'chunk_*.hdf5' | wc -l | tr -d ' '; }

job_running() {
    ssh -o ConnectTimeout=20 "$HOST" "squeue -j $JOBID -h 2>/dev/null | grep -q ." \
        >/dev/null 2>&1
}

analyse() {
    local n=$1 out
    out=$LOCAL/${RUNTAG}_$(date +%Y%m%d_%H%M)_${n}chunks
    $PYTHON "$HERE/run_analysis.py" --data "$DEST" --keep "$KEEP" --out "$out" \
        > "$out.log" 2>&1
    local seeds hdr
    seeds=$($PYTHON -c "
import glob, h5py
print(sum(len(h5py.File(f)['seed']) for f in glob.glob('$DEST/chunk_*.hdf5')))
" 2>/dev/null | tail -1)
    # Carry the headline numbers in the line itself, so a notification says what
    # the data now shows rather than only where to go and read it.
    # The quantity name is one to three words, so the shift is found as the first
    # field that looks like a signed number rather than by counting columns.
    hdr=$(awk '/^(large-scale power|tidal shear|knot fraction|void fraction|mean overdensity)/ {
                   name = ""; val = "";
                   for (i = 1; i <= NF; i++)
                       if ($i ~ /^[+-][0-9]/) { val = $i; break }
                       else name = name $i " ";
                   if (val != "") printf "%s%s  ", name, val }' \
              "$out/shifts_raw_theory.txt" 2>/dev/null | head -c 400)
    echo "$seeds realizations, $n chunks -> ${out##*/}"
    [ -n "$hdr" ] && echo "    $hdr"
}

last=0
while true; do
    # Quoted so the glob is expanded on the far side, not here.
    rsync -q --ignore-existing "$HOST:$REMOTE/data/$RUNTAG/chunk_*.hdf5" "$DEST/" \
        2>/dev/null || true
    n=$(count_local)

    if [ "$n" -ge $(( last + STEP )) ] && [ "$n" -gt 0 ]; then
        analyse "$n"
        last=$n
    fi

    if ! job_running; then
        # One last pass, so the final chunks are not left unanalysed.
        rsync -q --ignore-existing "$HOST:$REMOTE/data/$RUNTAG/chunk_*.hdf5" "$DEST/" \
            2>/dev/null || true
        n=$(count_local)
        [ "$n" -gt "$last" ] && analyse "$n"
        echo "job $JOBID has left the queue; $n chunks in total"
        break
    fi

    sleep "$INTERVAL"
done
