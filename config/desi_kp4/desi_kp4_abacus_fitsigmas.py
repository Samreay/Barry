import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import scipy as sp
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=500)

    mocktypes = ["abacus_cubicbox", "abacus_cutsky"]
    nzbins = [1, 3]

    # Loop over the mocktypes
    allnames = []
    for i, (mocktype, redshift_bins) in enumerate(zip(mocktypes, nzbins)):

        # Loop over the available redshift bins for each mock type
        for z in range(redshift_bins):

            # Loop over pre- and post-recon power spectrum measurements
            for recon in [None, "sym"]:

                # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
                # First load up mock mean and add it to the fitting list.
                dataset = PowerSpectrum_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_k=0.02,
                    max_k=0.30,
                    mocktype=mocktype,
                    redshift_bin=z + 1,
                    realisation=None,
                    num_mocks=1000,
                    reduce_cov_factor=25,
                )

                model = PowerBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    fix_params=["om", "alpha", "epsilon", "sigma_s"],
                    marg="full",
                    poly_poles=dataset.fit_poles,
                    correction=Correction.HARTLAP,
                    n_poly=5,
                )
                model.set_default("sigma_s", 0.0)

                name = dataset.name + " mock mean sigma_s=0"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

                dataset = CorrelationFunction_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_dist=52.0,
                    max_dist=150.0,
                    mocktype=mocktype,
                    redshift_bin=z + 1,
                    realisation=None,
                    num_mocks=1000,
                    reduce_cov_factor=25,
                )

                model = CorrBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    marg="full",
                    fix_params=["om", "beta", "alpha", "epsilon", "sigma_s"],
                    poly_poles=dataset.fit_poles,
                    correction=Correction.HARTLAP,
                )
                model.set_default("beta", 0.4)
                model.set_default("sigma_s", 0.0)

                model2 = CorrBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    marg="full",
                    fix_params=["om", "alpha", "b2", "epsilon", "sigma_s"],
                    poly_poles=dataset.fit_poles,
                    correction=Correction.HARTLAP,
                )
                model.set_default("beta", 0.4)
                model.set_default("sigma_s", 0.0)

                # Create a unique name for the fit and add it to the list
                name = dataset.name + " mock mean sigma_s=0"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

    # Submit all the job. We have quite a few (52), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below here is for plotting the chains once they have been run. The should_plot()
    # function will check for the presence of chains and plot if it finds them on your laptop. On the HPC you can
    # also force this by passing in "plot" as the second argument when calling this code from the command line.
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        # Set up a ChainConsumer instance. Plot the MAP for individual realisations and a contour for the mock average
        fitname = []
        c = [
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
            ChainConsumer(),
        ]

        zmins = ["0.4", "0.6", "0.8"]

        # Loop over all the chains
        stats = {}
        output = {}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            if "sigma_s" not in extra["name"]:
                continue

            # Get the realisation number and redshift bin
            print(extra["name"].split("_")[0])
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            if "z" in extra["name"].split("_")[1]:
                redshift_bin = [recon_bin + 2 * (i + 1) for i, zmin in enumerate(zmins) if zmin in extra["name"].split("_")[1]][0]
            else:
                redshift_bin = recon_bin

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior.argmax()
            params = df.loc[max_post]
            params_dict = model.get_param_dict(chain[max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)

            # Get some useful properties of the fit, and plot the MAP model against the data if it's the mock mean
            figname = "/".join(pfn.split("/")[:-1]) + "/" + extra["name"].replace(" ", "_") + "_bestfit.png"
            new_chi_squared, dof, bband, mods, smooths = model.plot(params_dict, display=False, figname=figname)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if "Pk" in extra["name"]:
                extra["name"] = "Pk"
                fitname.append(data[0]["name"].replace(" ", "_"))
            else:
                extra["name"] = "Xi"
            c[redshift_bin].add_chain(df, weights=weight, **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)
            mean, cov = weighted_avg_and_cov(
                df[["$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$"]],
                weight,
                axis=0,
            )
            print(redshift_bin, mean, np.sqrt(np.diag(cov)), new_chi_squared, dof)

        for redshift_bin in range(len(c)):
            c[redshift_bin].configure(bins=20)
            c[redshift_bin].plotter.plot(
                filename=["/".join(pfn.split("/")[:-1]) + "/" + fitname[redshift_bin] + "_contour.png"],
                parameters=["$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$"],
                legend=True,
            )
            # print(fitname[recon_bin], c[recon_bin].analysis.get_summary())
