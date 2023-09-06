import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import Optimiser
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.

# Convenience function to plot histograms of the errors and cross-correlation coefficients
def plot_errors(stats, figname, type="xi"):

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    if type == "xi":
        colmin = 1
        colmax = 6
    else:
        colmin = 3
        colmax = 7

    statsmean = np.mean(stats, axis=0)
    statsstd = np.std(stats, axis=0)

    fig, axes = plt.subplots(figsize=(4, 5), nrows=4, ncols=colmax - colmin, sharex=True, sharey="row", squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
    for n_poly in range(colmax - colmin):
        index = np.where(statsmean[:, 1] == n_poly + colmin)[0]
        print(statsmean[index, 0], statsmean[index, 2], statsmean[index, 3])

        for param in range(2):
            axes[param, n_poly].plot(
                statsmean[index, 0], statsmean[index, param + 2] * 100.0, color=colors[n_poly + colmin - 1], zorder=1, alpha=0.75, lw=0.8
            )
            axes[param, n_poly].fill_between(
                statsmean[index, 0],
                (statsmean[index, param + 2] - statsstd[index, param + 2]) * 100.0,
                (statsmean[index, param + 2] + statsstd[index, param + 2]) * 100.0,
                color=colors[n_poly + colmin - 1],
                zorder=1,
                alpha=0.5,
                lw=0.8,
            )
            axes[param, n_poly].axhline(0.0, color="k", ls="--", zorder=0, lw=0.8)
            axes[param + 2, n_poly].plot(
                statsmean[index, 0], statsstd[index, param + 2] * 100.0, color=colors[n_poly + colmin - 1], zorder=1, alpha=0.75, lw=0.8
            )

        axes[0, n_poly].set_ylim(-0.5 * 2.0, 0.5 * 2.0)
        axes[1, n_poly].set_ylim(-0.35 * 2.0, 0.35 * 2.0)
        # axes[2, n_poly].set_ylim(0.133 * 5.0, 0.150 * 5.0)
        # axes[3, n_poly].set_ylim(0.081 * 5.0, 0.091 * 5.0)
        if n_poly == int(np.floor(colmax - colmin) / 2.0):
            axes[3, n_poly].set_xlabel(r"$\sigma_{\Sigma}\,(h^{-1}\mathrm{Mpc})$", fontsize=12)
        if n_poly == 0:
            axes[0, n_poly].set_ylabel(r"$\alpha_{||} - 1\,(\%)$")
            axes[1, n_poly].set_ylabel(r"$\alpha_{\perp} - 1\,(\%)$")
            axes[2, n_poly].set_ylabel(r"$\sigma_{\alpha_{||}}\,(\%)$")
            axes[3, n_poly].set_ylabel(r"$\sigma_{\alpha_{\perp}}\,(\%)$")
        axes[0, n_poly].text(
            0.05,
            0.95,
            f"$N_{{poly}} = {{{n_poly+colmin}}}$",
            transform=axes[0, n_poly].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=colors[n_poly + colmin - 1],
        )

    fig.savefig(figname, bbox_inches="tight", dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = Optimiser(temp_dir=dir_name)

    sigma = {None: [9.6, 4.8, 2.0], "sym": [5.1, 1.6, 0.0]}
    sigma_sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon measurements
    for r, recon in enumerate([None, "sym"]):

        # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
        # First load up mock mean and add it to the fitting list.
        dataset_pk = PowerSpectrum_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=1,
            datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
        )

        dataset_xi = CorrelationFunction_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_dist=52.0,
            max_dist=150.0,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=1,
            datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
        )

        # Loop over pre- and post-recon measurements
        for sig in range(len(sigma_sigma)):

            for n_poly in range(3, 7):

                model = PowerBeutler2017(
                    recon=dataset_pk.recon,
                    isotropic=dataset_pk.isotropic,
                    fix_params=["om"],
                    marg="full",
                    poly_poles=dataset_pk.fit_poles,
                    correction=Correction.NONE,
                    n_poly=n_poly,
                )
                model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")
                model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")
                model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.kvals, model.pksmooth, model.pkratio = pktemplate.T

                dataset_pk.set_realisation(None)
                name = dataset_pk.name + f" mock mean fixed_type {sig} n_poly=" + str(n_poly)
                fitter.add_model_and_dataset(model, dataset_pk, name=name, color=colors[sig])
                allnames.append(name)

                for j in range(len(dataset_pk.mock_data)):
                    dataset_pk.set_realisation(j)
                    name = dataset_pk.name + f" realisation {j} fixed_type {sig} n_poly=" + str(n_poly)
                    fitter.add_model_and_dataset(model, dataset_pk, name=name, color=colors[sig])
                    allnames.append(name)

            for n_poly in range(1, 6):

                model = CorrBeutler2017(
                    recon=dataset_xi.recon,
                    isotropic=dataset_xi.isotropic,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=dataset_xi.fit_poles,
                    correction=Correction.NONE,
                    n_poly=n_poly,
                )
                model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")
                model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")
                model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=sigma_sigma[sig], prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                dataset_xi.set_realisation(None)
                name = dataset_xi.name + f" mock mean fixed_type {sig} n_poly=" + str(n_poly)
                fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[sig])
                allnames.append(name)

                for j in range(len(dataset_xi.mock_data)):
                    dataset_xi.set_realisation(j)
                    name = dataset_xi.name + f" realisation {j} fixed_type {sig} n_poly=" + str(n_poly)
                    fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[sig])
                    allnames.append(name)

    # Submit all the jobs to NERSC. We have quite a few (72), so we'll
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
        datanames = ["Xi_CV", "Pk_CV"]

        # Loop over all the chains
        stats = [[[] for _ in range(len(dataset_xi.mock_data))] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            if "Prerecon" in extra["name"]:
                continue

            if "mock mean" in extra["name"]:
                continue

            # Get the realisation number and redshift bin
            data_bin = 0 if "Xi" in extra["name"] else 1
            sigma_bin = int(extra["name"].split("fixed_type ")[1].split(" ")[0])
            realisation = int(extra["name"].split("realisation ")[1].split(" ")[0])

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels()).to_numpy()[0]

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df[0], df[1])

            stats[data_bin][realisation].append(
                [
                    sigma_sigma[sigma_bin],
                    model.n_poly,
                    alpha_par - 1.0,
                    alpha_perp - 1.0,
                    df[2] - 5.1,
                    df[3] - 1.6,
                    df[4],
                ]
            )

        print(np.array(np.shape(stats[1])))

        # Plot the error on the alpha parameters as a function of the width of the sigma prior
        plot_errors(np.array(stats[0]), "/".join(pfn.split("/")[:-1]) + "/" + datanames[0] + "_alphas.png", type="xi")
        plot_errors(np.array(stats[1]), "/".join(pfn.split("/")[:-1]) + "/" + datanames[1] + "_alphas.png", type="pk")
