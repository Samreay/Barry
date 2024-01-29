import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import NautilusSampler
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
    sampler = NautilusSampler(temp_dir=dir_name)

    sigma = {None: [9.5, 5.0, 2.0], "sym": [5.0, 2.0, 2.0]}

    # colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon power spectrum measurements
    for recon in ["sym"]:

        # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
        # First load up mock mean and add it to the fitting list.
        dataset_pk = PowerSpectrum_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=25,
            datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
        )

        for n, marg in enumerate([None, "partial", "full"]):

            model = PowerBeutler2017(
                recon=dataset_pk.recon,
                isotropic=dataset_pk.isotropic,
                fix_params=["om"],
                marg=marg,
                poly_poles=dataset_pk.fit_poles,
                correction=Correction.HARTLAP,
                n_poly=8,
            )
            model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
            model.set_default("beta", 0.4, min=0.1, max=0.7)
            model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
            model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

            # Load in a pre-existing BAO template
            pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
            model.kvals, model.pksmooth, model.pkratio = pktemplate.T

            name = dataset_pk.name + f" mock mean marg=" + str(n)
            fitter.add_model_and_dataset(model, dataset_pk, name=name)
            allnames.append(name)

    # Submit all the job. We have quite a few (42), so we'll
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
        c = ChainConsumer()
        marg_names = ["No\,Analytic\,Marginalisation", "Partial\,Analytic\,Marginalisation", "Full\,Analytic\,Marginalisation"]

        # Loop over all the chains
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            marg_bin = int(extra["name"].split("marg=")[1].split(" ")[0])
            print(extra["name"], marg_bin)

            # Store the chain in a dictionary with parameter names
            print(np.shape(chain), model.get_labels(), model.marg)
            print(model.get_active_params())
            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            df["$\\alpha_{ap}$"] = (1.0 + df["$\\epsilon$"].to_numpy()) ** 3
            print(df.keys())

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
            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(params_dict, display=False, figname=figname)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            extra["name"] = marg_names[marg_bin]
            c.add_chain(df, weights=weight, **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)

        truth = {
            "$\\alpha$": 1.0,
            "$\\alpha_{ap}$": 1.0,
            "$\\alpha_\\perp$": 1.0,
            "$\\alpha_\\parallel$": 1.0,
            "$\\Sigma_{nl,||}$": sigma["sym"][0],
            "$\\Sigma_{nl,\\perp}$": sigma["sym"][1],
            "$\\Sigma_s$": sigma["sym"][2],
        }
        c.plotter.plot(
            legend=True,
            truth=truth,
            filename="/".join(pfn.split("/")[:-1]) + "/marg_contour.png",
        )
        c.plotter.plot(
            parameters=[
                "$\\alpha$",
                "$\\alpha_{ap}$",
                "$b_{0,1}$",
                "$\\beta$",
                "$\\Sigma_{nl,||}$",
                "$\\Sigma_{nl,\\perp}$",
                "$\\Sigma_s$",
            ],
            legend=True,
            truth=truth,
            filename="/".join(pfn.split("/")[:-1]) + "/marg_contour2.png",
        )
