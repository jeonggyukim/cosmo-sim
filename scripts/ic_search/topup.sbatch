#!/bin/bash
# Rerun the seeds a sweep did not measure, listed one per line in $SEEDFILE.
#
# Build the list first with collect_missing.py, which finds both the seeds
# monofonIC refused to generate and the seeds whose task never wrote a chunk:
#
#   python collect_missing.py --data $W/data/big128 \
#          --expect-start 100000 --expect-count 100000
#   sbatch --array=0-19 --export=ALL,SEEDFILE=$W/data/big128/missing_seeds.txt,\
#          NPER=100,RUNTAG=big128_topup topup.sbatch
#
# Size the array so that NPER x (array width) covers the list; a task whose
# slice starts past the end of the list exits without doing anything, so
# overshooting is harmless.
#
# The results go in their own RUNTAG directory rather than into the original
# one, because the original's chunk files record which seeds they skipped and
# rewriting that history would make the run harder to account for. Point the
# analysis at both directories, or merge them.

#SBATCH --job-name=topup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/jeonggyukim/monofonic-tests/logs/%x_%A_%a.out

set -eo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gnu12/12.2.0 openmpi4/4.1.5 fftw/3.3.10 hdf5/1.14.0 gsl/2.7.1

: "${WORKDIR:=/gpfs/jeonggyukim/monofonic-tests}"
: "${PYTHON:=$HOME/miniconda3/envs/pyathena/bin/python}"
: "${NPER:=100}"
: "${NGRID:=128}"
: "${FLUSH:=25}"
: "${BATCH:=25}"
: "${RUNTAG:=topup}"
: "${SEEDFILE:?set SEEDFILE to the list written by collect_missing.py}"

export MONOFONIC_TESTS=$WORKDIR
export MONOFONIC_REF=$WORKDIR/ref
export MONOFONIC_BIN=$WORKDIR/build/monofonIC
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

START=$(( SLURM_ARRAY_TASK_ID * NPER ))
OUT=$WORKDIR/data/$RUNTAG
mkdir -p "$OUT" "$WORKDIR/logs"

cd "$WORKDIR/scripts"
exec $PYTHON pencil_seed_sweep.py \
    --seed-list "$SEEDFILE" --list-start "$START" --nseeds "$NPER" --ngrid "$NGRID" \
    --species matter --npencils 24 --environment --smooth 20 40 \
    --nthreads 1 --compact --flush-every "$FLUSH" --batch-seeds "$BATCH" \
    --out "$OUT"
