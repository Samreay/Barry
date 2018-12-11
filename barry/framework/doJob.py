import os
import shutil
import logging


def write_jobscript_slurm(filename, name=None, num_tasks=24, num_cpu=24,
                          delete=False, partition="smp", conda_env="sam35"):

    directory = os.path.dirname(os.path.abspath(filename))
    executable = os.path.basename(filename)
    if name is None:
        name = executable[:-3]
    output_dir = directory + os.sep + "out_files"
    if delete and os.path.exists(output_dir):
        logging.debug("Deleting %s" % output_dir)
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    template = f'''#!/bin/bash -l
#SBATCH -p {partition}
#SBATCH -J {name}
#SBATCH --array=1-{num_tasks}%{num_cpu}
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -o {output_dir}/{name}.o%j

IDIR={directory}
conda deactivate
conda activate {conda_env}
echo $PATH
echo "Activated python"
executable=$(which python)
echo $executable

PROG={executable}
PARAMS=`expr ${{SLURM_ARRAY_TASK_ID}} - 1`
cd $IDIR
sleep $((RANDOM % 10))
srun -N 1 -n 1 -c 1 $executable $PROG $PARAMS'''

    n = "%s/%s.q" % (directory, executable[:executable.index(".py")])
    with open(n, 'w') as f:
        f.write(template)
    logging.info("SLURM Jobscript at %s" % n)
    return n
