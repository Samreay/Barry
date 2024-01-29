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

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon power spectrum measurements
    for recon in ["sym"]:
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

        dataset_xi = CorrelationFunction_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_dist=52.0,
            max_dist=150.0,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=25,
            datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
        )
        for fog_wiggles in [True, False]:
            for dilate_smooth in [True, False]:

                model = PowerBeutler2017(
                    recon=dataset_pk.recon,
                    isotropic=dataset_pk.isotropic,
                    fix_params=["om"],
                    marg="full",
                    poly_poles=dataset_pk.fit_poles,
                    correction=Correction.HARTLAP,
                    fog_wiggles=fog_wiggles,
                    dilate_smooth=dilate_smooth,
                )
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
                model.set_default("beta", 0.4, min=0.1, max=0.7)

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.kvals, model.pksmooth, model.pkratio = pktemplate.T

                name = dataset_pk.name + " mock mean"
                fitter.add_model_and_dataset(model, dataset_pk, name=name)
                allnames.append(name)

                model = CorrBeutler2017(
                    recon=dataset_xi.recon,
                    isotropic=dataset_xi.isotropic,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=dataset_xi.fit_poles,
                    correction=Correction.HARTLAP,
                    fog_wiggles=fog_wiggles,
                    dilate_smooth=dilate_smooth,
                    n_poly=[0, 2],
                )
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                model.set_default("beta", 0.4, min=0.1, max=0.7)

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                name = dataset_xi.name + " mock mean"
                fitter.add_model_and_dataset(model, dataset_xi, name=name)
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
        c = [
            ChainConsumer(),
            ChainConsumer(),
        ]
        fitname = [None for i in range(len(c))]

        # Loop over all the chains
        stats = {}
        output = {}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            poly_bin = int(extra["name"].split("n_poly=")[1].split(" ")[0])
            data_bin = 0 if "Xi" in extra["name"] else 1
            print(extra["name"], recon_bin, data_bin)

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
            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(params_dict, display=False, figname=figname)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if data_bin == 0 and poly_bin == 0:
                fitname[recon_bin] = data[0]["name"].replace(" ", "_")
            chainname = [r"$\xi(s)$" if data_bin == 0 else r"$P(k)$", "Polynomial" if poly_bin == 0 else r"Spline"]
            extra["name"] = chainname[0] + " " + chainname[1]
            c[recon_bin].add_chain(df, weights=weight, **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)
            mean, cov = weighted_avg_and_cov(
                df[["$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"]],
                weight,
                axis=0,
            )

        for recon_bin in range(len(c)):
            if "Pre" in fitname[recon_bin]:
                truth = {"$\\Sigma_{nl,||}$": 9.6, "$\\Sigma_{nl,\\perp}$": 4.8, "$\\Sigma_s$": 2.0}
            else:
                truth = {"$\\Sigma_{nl,||}$": 5.1, "$\\Sigma_{nl,\\perp}$": 1.6, "$\\Sigma_s$": 0.0}
            c[recon_bin].configure(bins=20, sigmas=[0, 1])
            fig = c[recon_bin].plotter.plot(
                parameters=["$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"],
                legend=True,
                truth=truth,
            )
            xvals = np.linspace(0.0, 20.0, 100)
            fig.get_axes()[3].plot(xvals, xvals / (1.0 + 0.8), color="k", linestyle=":", linewidth=1.3)
            fig.savefig(
                fname="/".join(pfn.split("/")[:-1]) + "/" + fitname[recon_bin] + "_contour.png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.05,
            )
