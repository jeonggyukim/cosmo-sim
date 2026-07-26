#!/bin/bash
# monofonic.sbatch — multi-node monofonIC IC generation via run-pipeline.
#
# The batch script runs on one node; run-pipeline (the Python orchestrator) runs
# there too and launches only the monofonIC binary across the whole allocation
# with `srun` (via --launcher). Edit REPO, the module names, and the SBATCH sizes
# for your site (`module avail`), then: sbatch tools/monofonic.sbatch
#
#SBATCH --job-name=monofonic-ic
#SBATCH --partition=normal
#SBATCH --output=monofonic-%j.out
#SBATCH --error=monofonic-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # MPI ranks per node ...
#SBATCH --cpus-per-task=48           # ... x OpenMP threads per rank
#SBATCH --exclusive

# --- Environment (adjust module names to your site) ---
module purge
module load gnu12/12.2.0
module load openmpi4/4.1.5
module load fftw/3.3.10               # must provide libfftw3_mpi
module load hdf5/1.14.0               # parallel HDF5
module load gsl/2.7.1
module load cmake/4.0.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Optional OpenMPI transport tweaks for sites where UCX/openib misbehaves:
# export OMPI_MCA_pml=ob1
# export OMPI_MCA_btl=tcp,self,vader

REPO=/path/to/cosmo-sim          # <-- edit
cd "$REPO"
conda activate cosmo                  # or: source activate cosmo

# Build once (skips if the binary exists). MPI is required for monofonIC.
tools/build-monofonic.sh

# Generate the config and run monofonIC across all ranks. --launcher wraps only
# the IC binary in `srun`, which inherits this job's node/task allocation;
# --stop-after-ic skips the single-node analysis steps.
run-pipeline --ic-code monofonic \
    --ngrid 1024 --lbox 1000 --zstart 200 \
    --nthreads "$SLURM_CPUS_PER_TASK" \
    --launcher "srun" \
    --stop-after-ic

# Downstream validation is single-node — run it separately, only if the density
# grid fits one node's memory (otherwise use an MPI P(k)/xi estimator):
#   run-pipeline --ic-code monofonic --ngrid 1024 --lbox 1000 --zstart 200
