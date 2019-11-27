import inspect
import os
import shutil
import logging
from barry.config import get_config


def write_jobscript_slurm(filename, name=None, num_tasks=24, num_concurrent=24, delete=False):
    config = get_config()
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

    d = {"directory": directory, "executable": executable, "name": name, "output_dir": output_dir, "num_concurrent": num_concurrent, "num_tasks": num_tasks}
    d.update(config)

    slurm_job = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), "jobscripts/slurm_fit.job")
    with open(slurm_job) as f:
        raw_template = f.read()
    template = raw_template.format(**d)

    n = "%s/%s.q" % (q_dir, executable[: executable.index(".py")])
    with open(n, "w") as f:
        f.write(template)
    logging.info("SLURM Jobscript at %s" % n)
    return n
