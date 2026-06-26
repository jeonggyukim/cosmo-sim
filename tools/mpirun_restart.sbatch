#!/bin/bash
#SBATCH --job-name=CV22music
#SBATCH --partition=normal
#SBATCH --output=output-%j.txt
#SBATCH --error=error-%j.txt
#SBATCH --time=168:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --exclude=grammar008,grammar038,grammar035,grammar066,grammar048,grammar045,grammar026
#SBATCH --exclusive
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=128M 
# Your commands go here
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
export LIBRARY_PATH=$LIBRARY_PATH:/home/USER/local/GKlib/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home/USER/local/METIS/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home/USER/local/ParMETIS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/USER/local/GKlib/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/USER/local/METIS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/USER/local/ParMETIS/lib


echo "=== 清理所有 UCX 和冲突的环境变量 ==="
for var in $(env | grep UCX | cut -d= -f1); do
    unset $var
done

echo "=== 设置纯 TCP MPI 配置 ==="
export OMPI_MCA_pml=ob1
export OMPI_MCA_pml_ucx_enable=0          # 明确禁用 UCX
export OMPI_MCA_btl=tcp,self,vader
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0
export OMPI_MCA_btl=^openib
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_plm_rsh_num_concurrent=768
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_max_send_size=131072

date


# 简单启用 core dump
ulimit -c unlimited
ulimit -s unlimited
cd /path/to/run/

mpirun -np 96  /path/to/run/swift_mpi --threads=4 --pin\
    --cosmology --eagle --verbose=1 \
    CV_23_restart.yml

