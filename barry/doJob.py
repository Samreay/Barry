import os
import shutil
import logging

from barry.config import get_config


def write_jobscript_slurm(filename, name=None, num_tasks=24, num_cpu=24,
                          delete=False, partition="smp"):
    config = get_config()
    conda_env = config["conda_env"]
    directory = os.path.dirname(os.path.abspath(filename))
    executable = os.path.basename(filename)
    if name is None:
        name = executable[:-3]
    output_dir = directory + os.sep + "out_files"
    q_dir = directory + os.sep + "job_files"
    if not os.path.exists(q_dir):
        os.makedirs(q_dir, exist_ok=True)
    if delete and os.path.exists(output_dir):
        logging.debug("Deleting %s" % output_dir)
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    template = f'''#!/bin/bash -l
#SBATCH -p {partition}
#SBATCH -J {name}
#SBATCH --array=1-{num_tasks}%{num_cpu}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=6G
#SBATCH -t 24:00:00
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
sleep $((RANDOM % 5))
$executable $PROG $PARAMS
'''

    n = "%s/%s.q" % (q_dir, executable[:executable.index(".py")])
    with open(n, 'w') as f:
        f.write(template)
    logging.info("SLURM Jobscript at %s" % n)
    return n
