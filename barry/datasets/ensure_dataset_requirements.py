import inspect
import sys
import os
import logging


sys.path.append("../..")
from barry.config import is_local, get_config
from barry.cosmology.camb_generator import CambGenerator
from barry.datasets.dataset import Dataset
from tests.utils import get_concrete


def setup_ptgenerator_slurm(c, launched):
    config = get_config()
    job_path = os.path.join(os.path.dirname(inspect.stack()[0][1]), "..", "jobscripts/slurm_pt_generator.job")

    d = {"partition": config["job_partition"], "conda_env": config["job_conda_env"], "mpi_module": config["mpi_module"]}
    with open(job_path) as f:
        raw_template = f.read()
    d.update(c)
    template = raw_template.format(**d)

    job_dir = "jobs"
    unique_name = "".join([k + str(c[k]) for k in sorted(c.keys())]) + ".job"
    filename = os.path.join(job_dir, unique_name)
    os.makedirs(job_dir, exist_ok=True)
    with open(filename, "w") as f:
        f.write(template)
    if filename not in launched:
        logging.info(f"Submitting {filename}")
        os.system(f"{config['hpc_submit_command']} {filename}")
        launched.append(filename)


def ensure_requirements(dataset, launched):
    for data in dataset.get_data():
        c = data["cosmology"]

        # Ensure we have CAMB pre-generated. This can be done locally as it is fast.
        generator = CambGenerator(om_resolution=101, h0_resolution=1, h0=c["h0"], ob=c["ob"], ns=c["ns"], redshift=c["z"])
        generator.load_data(can_generate=True)

        setup_ptgenerator_slurm(c, launched)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")

    # This should be run on a HPC for the PTGenerator side of things.
    assert not is_local(), "Please run this on your HPC system"

    classes = get_concrete(Dataset)
    concrete = [c() for c in classes]
    launched = []
    for dataset in concrete:
        logging.info(f"Ensuring requirements for {dataset.name}")
        ensure_requirements(dataset, launched)
