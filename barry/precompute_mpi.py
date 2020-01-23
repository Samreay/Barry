import sys
import logging
import argparse
import numpy as np

sys.path.insert(0, "..")
from barry.models import Model
from tests.utils import get_concrete


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")

    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--redshift", type=float, default=0.51)
    parser.add_argument("--om", type=float, default=0.31)
    parser.add_argument("--h0", type=float, default=0.676)
    parser.add_argument("--ob", type=float, default=0.04814)
    parser.add_argument("--ns", type=float, default=0.97)
    parser.add_argument("--reconsmoothscale", type=float, default=21.21)
    args = parser.parse_args()

    assert args.model is not None, "This file is invoked by generate.py and requires you to pass in a model name, redshift, om, h0, ob, ns and reconsmoothscale"
    assert len(get_concrete(Model)) > 0, "get_concrete(Model) reports no subclasses. Send Sam and email, imports are funky."

    # Find the right model
    model = [c() for c in get_concrete(Model) if args.model == c.__name__][0]
    logging.info(f"Model found is {model}")
    model.set_cosmology(
        {"z": args.redshift, "h0": args.h0, "om": args.om, "ob": args.ob, "ns": args.ns, "reconsmoothscale": args.reconsmoothscale}, load_pregen=False
    )

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    logging.info(f"Process reporting rank {rank} and size {size}")

    # Ensure that the camb data loads
    if rank == 0:
        model.camb.load_data(can_generate=True)
    mpi_comm.Barrier()

    camb = model.camb
    omch2s = camb.omch2s
    h0s = camb.h0s
    all_indexes = [(i, j) for i in range(len(omch2s)) for j in range(len(h0s))]

    assert mpi_comm is not None, "Yeah this is going to need MPI, which isnt working for some reason..."
    size = mpi_comm.Get_size()
    logging.info(f"Running generation via MPI with size {size}")

    # Split the tasks based on how many MPI cores we have
    delegations = [all_indexes[i::size] for i in range(size)]
    run_indexes = mpi_comm.scatter(delegations, root=0)
    results = model.generate_precomputed_data(run_indexes)
    all_results = mpi_comm.gather(results, root=0)

    # Should have a list of results mapping indexes to values that have come back. Unpack these into fixed arrays
    if rank == 0:
        logging.info("Merging results")
        data = all_results[0]  # Start with the first set

        # Use the shapes to create the right sized arrays
        tmp = data[0][2]  # This is the first thing model.precompute would have generated
        combined = {}
        num = []
        for key in tmp.keys():
            if isinstance(tmp[key], (float, int)):
                combined[key] = np.empty((model.camb.om_resolution, model.camb.h0_resolution))
                num.append(key)
            else:
                combined[key] = np.empty((model.camb.om_resolution, model.camb.h0_resolution, len(tmp[key])))
            combined[key][:] = np.nan

        # Combine the results of all different cores
        for data in all_results:
            for i, j, values in data:
                for k, v in values.items():
                    if k in num:
                        combined[k][i, j] = v
                    else:
                        combined[k][i, j, :] = v

        model._save_precomputed_data(combined)
