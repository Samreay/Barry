#!/bin/bash -l
#SBATCH -J {name}
#SBATCH --qos=shared
#SBATCH -C haswell
#SBATCH --ntasks=20
#SBATCH --account={account}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 04:00:00
#SBATCH -n 20
#SBATCH -o {output}.o%j

source ~/.bashrc.ext
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
echo "Activated python"
echo `which python`

cd {path}
mpirun python precompute_mpi.py --model {model} --reconsmoothscale {reconsmoothscale} --redshift {z} --om {om} --h0 {h0} --ob {ob} --ns {ns} --mnu {mnu}
