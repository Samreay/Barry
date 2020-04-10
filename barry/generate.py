import inspect
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.abspath(".."))
from barry.models import Model
from barry.config import is_local, get_config
from barry.cosmology.camb_generator import CambGenerator
from barry.datasets.dataset import Dataset
from tests.utils import get_concrete


def setup_ptgenerator_slurm(model, c):
    config = get_config()
    job_path = os.path.join(os.path.dirname(inspect.stack()[0][1]), "jobscripts/slurm_pt_generator.job")
    python_path = os.path.abspath(os.path.dirname(inspect.stack()[0][1]))
    unique_name = model.__class__.__name__ + "_" + ("".join([k + str(c[k]) for k in sorted(c.keys())])) + ".job"
    job_dir = os.path.abspath("jobs")
    output = os.path.join(job_dir, "zlog")
    d = {
        "partition": config["job_partition"],
        "name": unique_name,
        "conda_env": config["job_conda_env"],
        "mpi_module": config["mpi_module"],
        "fort_compile_module": config["fort_compile_module"],
        "path": python_path,
        "output": output,
        "model": model.__class__.__name__,
    }
    with open(job_path) as f:
        raw_template = f.read()
    d.update(c)
    template = raw_template.format(**d)

    filename = os.path.join(job_dir, unique_name)
    os.makedirs(job_dir, exist_ok=True)
    with open(filename, "w") as f:
        f.write(template)
    logging.info(f"Submitting regen for {filename}")
    os.system(f"{config['hpc_submit_command']} {filename}")


def get_cosmologies(datasets):
    # This is an annoying hack because the dicts are ==, but have different ids, so cannot use is in
    cs = []
    for ds in datasets:
        for d in ds.get_data():
            c = d["cosmology"]
            found = False
            for c2 in cs:
                if c == c2:
                    found = True
            if not found:
                cs.append(c)
    return cs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s |%(funcName)20s]   %(message)s")

    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--refresh", action="store_true", default=False)
    args = parser.parse_args()

    # This should be run on a HPC for the PTGenerator side of things.
    assert not is_local(), "Please run this on your HPC system"

    datasets = [c(realisation="data") for c in get_concrete(Dataset) if "DESI" in c.__name__]

    cosmologies = get_cosmologies(datasets)
    logging.info(f"Have {len(cosmologies)} cosmologies")

    # Ensure all cosmologies exist
    for c in cosmologies:
        logging.info(f"Ensuring cosmology {c} is generated")
        generator = CambGenerator(om_resolution=101, h0_resolution=1, h0=c["h0"], ob=c["ob"], ns=c["ns"], redshift=c["z"], mnu=c["mnu"])
        generator.load_data(can_generate=True)

    # For each cosmology, ensure that each model pregens the right data
    models = [c() for c in get_concrete(Model)]
    for m in models:
        for c in cosmologies:
            try:
                m.set_cosmology(c)
                logging.info(f"Model {m.__class__.__name__} already has pregenerated data for {m.camb.filename_unique}")
                if args.refresh:
                    logging.info("But going to refresh tme anyway!")
                    assert not args.refresh, "Refreshing anyway!"
            except AssertionError:
                setup_ptgenerator_slurm(m, c)
