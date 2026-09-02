#!/bin/bash
# Seed sweep as a Slurm job array, one chunk of seeds per array task.
#
# Realizations share nothing, so this is embarrassingly parallel: no MPI, one
# core per task, and the array width is set by how many cores are free rather
# than by the problem. Each task writes one self-contained chunk file, so tasks
# never read each other's output and a failed task can be resubmitted alone.
#
#   sbatch --array=0-399 --export=ALL,NPER=250,SEEDBASE=100000,RUNTAG=big sweep.sbatch
#
# gives 400 x 250 = 100,000 seeds. Seed ranges are derived from the task index,
# so they cannot overlap. Keep SEEDBASE distinct between runs that will later be
# merged; merge_sweeps.py refuses duplicate seeds.
#
# Variables, all overridable through --export:
#   NPER      seeds per array task            (default 200)
#   SEEDBASE  first seed of task 0            (default 100000)
#   NGRID     cells per side                  (default 128)
#   FLUSH     seeds per chunk file            (default 25; 0 = one file at the end)
#   BATCH     seeds per monofonIC call        (default 25; 1 = one call per seed)
#   SMOOTHFRAC smoothing radii as fractions of the pencil width (default 0.1 0.25 0.5)
#   MARGIN    interior margin in smoothing radii (default 1.0)
#   RUNTAG    subdirectory under data/        (default run)
#   WORKDIR   root holding scripts/ ref/ build/ data/ logs/
#   PYTHON    interpreter with numpy and h5py
#
# Wall time is per task, not for the array. With BATCH > 1 the transfer function
# is computed once per batch rather than once per seed, which is most of the
# cost: 67 s/seed at N = 128 on grammar with BATCH = 1, and roughly a third of
# that above it. --time=08:00:00 leaves a wide margin for NPER = 200 either way.

#SBATCH --job-name=pencil
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/jeonggyukim/monofonic-tests/logs/%x_%A_%a.out

set -eo pipefail

# A batch shell is not a login shell, so the module function is not defined yet.
source /etc/profile.d/lmod.sh
module purge
module load gnu12/12.2.0 openmpi4/4.1.5 fftw/3.3.10 hdf5/1.14.0 gsl/2.7.1

: "${WORKDIR:=/gpfs/jeonggyukim/monofonic-tests}"
: "${PYTHON:=$HOME/miniconda3/envs/pyathena/bin/python}"
: "${NPER:=200}"
: "${SEEDBASE:=100000}"
: "${NGRID:=128}"
: "${FLUSH:=25}"
: "${BATCH:=25}"
: "${SMOOTHFRAC:=0.1 0.25 0.5}"
: "${MARGIN:=1.0}"
: "${RUNTAG:=run}"

export MONOFONIC_TESTS=$WORKDIR
export MONOFONIC_REF=$WORKDIR/ref
export MONOFONIC_BIN=$WORKDIR/build/monofonIC
# monofonIC is threaded but each task owns one core; the measurement is
# single-threaded numpy either way.
export OMP_NUM_THREADS=1
# stdout goes to a file, not a tty, so Python would buffer the progress lines
# until the task ends and a running task would look silent.
export PYTHONUNBUFFERED=1

SEED0=$(( SEEDBASE + SLURM_ARRAY_TASK_ID * NPER ))
OUT=$WORKDIR/data/$RUNTAG
mkdir -p "$OUT" "$WORKDIR/logs"

cd "$WORKDIR/scripts"
exec $PYTHON pencil_seed_sweep.py \
    --seed0 "$SEED0" --nseeds "$NPER" --ngrid "$NGRID" \
    --species matter --npencils 24 --environment \
    --smooth-frac $SMOOTHFRAC --interior-margin "$MARGIN" \
    --nthreads 1 --compact --flush-every "$FLUSH" --batch-seeds "$BATCH" \
    --out "$OUT"
