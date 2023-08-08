import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import Optimiser
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.fitter import Fitter

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Optimiser sampler.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = Optimiser(temp_dir=dir_name)

    # Specify the datafiles to fit
    datafiles = ["cubicbox_pk_lrg_abacussummit_c000_grid_c000.pkl", "cubicbox_pk_lrg_abacussummit_c000_grid_c001.pkl"]

    # Set up the models pre and post recon
    model_pre = PowerBeutler2017(
        recon=None,
        isotropic=False,
        marg="full",
        poly_poles=[0, 2],
        n_poly=5,
    )
    model_post = PowerBeutler2017(
        recon="sym",
        isotropic=False,
        marg="full",
        poly_poles=[0, 2],
        n_poly=5,
    )

    # Loop over the datafiles and fit each mock realisation in the pairs
    allnames = []
    for i, datafile in enumerate(datafiles):

        # Loop over pre- and post-recon measurements
        for recon in [None, "sym"]:

            # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
            # First load up mock mean and add it to the fitting list. Use only the diagonal parts
            # of the covariance matrix
            dataset = PowerSpectrum_DESI_KP4(
                recon=recon,
                fit_poles=[0, 2],
                min_k=0.02,
                max_k=0.30,
                realisation=None,
                num_mocks=1000,
                datafile=datafile,
            )

            model = model_pre if recon is None else model_post

            # Now add the individual realisations to the list
            for j in range(len(dataset.mock_data)):
                dataset.set_realisation(j)
                name = dataset.name + f" realisation {j}"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

    # Submit all the job. We have quite a few (52), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)
