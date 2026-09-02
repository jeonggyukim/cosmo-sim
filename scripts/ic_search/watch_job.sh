#!/bin/bash
# Report on a sweep running on the cluster, optionally waiting for it first.
#
# The waiting loop runs on the cluster inside one ssh connection rather than
# reconnecting on every poll, so it can check often without opening a login
# session each time. The cost is that a dropped connection ends the watch
# instead of being retried silently, so silence here is not evidence that the
# job is healthy.
#
#   watch_job.sh JOBID RUNTAG                 report once, now
#   watch_job.sh JOBID RUNTAG --first-chunk   wait for the first chunk file
#   watch_job.sh JOBID RUNTAG --done          wait for every task to leave the queue
#
# Interval between polls is INTERVAL seconds, default 10.

set -uo pipefail

JOBID=${1:?usage: watch_job.sh JOBID RUNTAG [--first-chunk|--done]}
RUNTAG=${2:?usage: watch_job.sh JOBID RUNTAG [--first-chunk|--done]}
MODE=${3:---once}
INTERVAL=${INTERVAL:-10}
HOST=${CLUSTER_HOST:-g}
REMOTE=${CLUSTER_DIR:-/gpfs/jeonggyukim/monofonic-tests}

case "$MODE" in
    --once)        WAIT_FOR="" ;;
    --first-chunk) WAIT_FOR="until ls \$W/data/$RUNTAG/chunk_*.hdf5 >/dev/null 2>&1; do sleep $INTERVAL; done" ;;
    --done)        WAIT_FOR="while squeue -j $JOBID -h 2>/dev/null | grep -q .; do sleep $INTERVAL; done" ;;
    *) echo "unknown mode: $MODE"; exit 1 ;;
esac

# Single quoted heredoc: the whole script is built here and evaluated on the far
# side, so $W and the command substitutions below run there, not locally.
ssh -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" "
W=$REMOTE
$WAIT_FOR

echo \"job $JOBID, run $RUNTAG, \$(date '+%H:%M:%S') on the cluster\"
echo
echo 'task states:'
sacct -j $JOBID -X -n -o State 2>/dev/null | sort | uniq -c | sed 's/^/  /'
echo
echo \"chunks written : \$(ls -1 \$W/data/$RUNTAG/chunk_*.hdf5 2>/dev/null | wc -l)\"
echo \"seeds measured : \$(grep -ah '^seed ' \$W/logs/*_${JOBID}_*.out 2>/dev/null | wc -l)\"
echo \"seeds skipped  : \$(grep -ah ': skipped' \$W/logs/*_${JOBID}_*.out 2>/dev/null | wc -l)\"
echo
echo 'batch generation (first task):'
grep -ah 'generated' \$W/logs/*_${JOBID}_0.out 2>/dev/null | head -3 | sed 's/^/  /'
echo
echo 'most recent seeds (first task):'
grep -ah '^seed ' \$W/logs/*_${JOBID}_0.out 2>/dev/null | tail -3 | sed 's/^/  /'
" 2>&1 | grep -vE "bind \[|channel_setup|Could not request"
