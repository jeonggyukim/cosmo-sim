#!/bin/bash
#SBATCH --job-name=CV22music
#SBATCH --partition=normal
#SBATCH --output=output-%j.txt
#SBATCH --error=error-%j.txt
#SBATCH --time=168:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
# Exclude known-bad nodes here if the site needs it, e.g.
#   #SBATCH --exclude=node001,node002
#SBATCH --exclusive

# --- Edit these three before submitting -----------------------------------
RUNDIR=/path/to/run                 # working directory holding the restart files
SWIFT_BIN=$RUNDIR/swift_mpi         # SWIFT binary built with MPI
PARTMETIS_PREFIX=$HOME/local        # holds GKlib/, METIS/, ParMETIS/ subdirs
RESTART_YML=CV_23_restart.yml
NRANKS=96
# --------------------------------------------------------------------------

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=128M

module purge
module load gnu12/12.2.0
module load hwloc/2.12.0
module load libfabric/1.18.0
module load openmpi4/4.1.5
module load cmake/4.0.0
module load fftw/3.3.10
module load gsl/2.7.1
module load hdf5/1.14.0
module unload ucx
module list

for lib in GKlib METIS ParMETIS; do
    export LIBRARY_PATH=$LIBRARY_PATH:$PARTMETIS_PREFIX/$lib/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PARTMETIS_PREFIX/$lib/lib
done

# Clear every UCX variable in the environment.  UCX and the plain TCP
# transport selected below cannot both be active; leftover UCX settings from
# the module environment make MPI_Init hang rather than fail.
for var in $(env | grep UCX | cut -d= -f1); do
    unset $var
done

# Force plain TCP for MPI.  This is a workaround for an unreliable
# InfiniBand path on this cluster, not a performance choice: it is slower
# than native transport and should be removed once the fabric is fixed.
export OMPI_MCA_pml=ob1
export OMPI_MCA_pml_ucx_enable=0
export OMPI_MCA_btl=tcp,self,vader
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0
export OMPI_MCA_btl=^openib
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_plm_rsh_num_concurrent=768
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_max_send_size=131072

date

ulimit -c unlimited
ulimit -s unlimited

cd "$RUNDIR"

mpirun -np $NRANKS "$SWIFT_BIN" --threads=$SLURM_CPUS_PER_TASK --pin \
    --cosmology --eagle --verbose=1 \
    "$RESTART_YML"
