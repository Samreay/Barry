import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import NautilusSampler
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
from scipy.stats import gaussian_kde
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.


# Convenience function to plot histograms of the errors and cross-correlation coefficients
def plot_errors(sigma_prior_factor, stats, figname):

    mosaic = """AB
                CD
                EF
                GH"""
    fig = plt.figure(layout="constrained")
    left, right = fig.subfigures(nrows=1, ncols=2)
    axxi = left.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "bottom": 0.1,
            "top": 0.95,
            "left": 0.1,
            "right": 0.5,
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )
    axpk = right.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "bottom": 0.1,
            "top": 0.95,
            "left": 0.55,
            "right": 0.95,
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )

    for data_bin, ax in enumerate([axxi, axpk]):
        c = "#ff7f0e" if data_bin == 0 else "#1f77b4"
        tracer = r"$\xi(s)$" if data_bin == 0 else r"$P(k)$"

        for sigma_bin, vals in enumerate([["A", "C", "E", "G"], ["B", "D", "F", "H"]]):
            statsmean = np.array(stats[data_bin][sigma_bin])
            for ind, (label, range) in enumerate(
                zip(
                    [
                        r"$\Delta \alpha_{\mathrm{iso}}\,(\%)$",
                        r"$\Delta \alpha_{\mathrm{ap}}\,(\%)$",
                        r"$\sigma_{\alpha_{\mathrm{iso}}}\,(\%)$",
                        r"$\sigma_{\alpha_{\mathrm{ap}}}\,(\%)$",
                    ],
                    [[-0.25, 0.25], [-0.65, 0.65], [0.0, 0.10], [0.1, 0.3]],
                )
            ):
                ax[vals[ind]].plot(sigma_prior_factor, statsmean[:, ind], color=c, zorder=1, alpha=0.75, lw=0.8)
                if ind < 2:
                    ax[vals[ind]].fill_between(
                        sigma_prior_factor,
                        statsmean[:, ind] - statsmean[:, ind + 2],
                        statsmean[:, ind] + statsmean[:, ind + 2],
                        color=c,
                        zorder=1,
                        alpha=0.5,
                        lw=0.8,
                    )

                ax[vals[ind]].set_ylim(range[0], range[1])
                if ind == 3:
                    ax[vals[ind]].set_xlabel(r"$\times\Sigma\,{\mathrm{prior}}$")
                else:
                    ax[vals[ind]].set_xticklabels([])
                if data_bin == 0 and sigma_bin == 0:
                    ax[vals[ind]].set_ylabel(label)
                else:
                    ax[vals[ind]].set_yticklabels([])
                if ind == 0:
                    for val, ls in zip([-0.1, 0.0, 0.1], [":", "--", ":"]):
                        ax[vals[ind]].axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
                if ind == 1:
                    for val, ls in zip([-0.2, 0.0, 0.2], [":", "--", ":"]):
                        ax[vals[ind]].axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
                ax[vals[ind]].axvline(1.0, color="k", ls=":", zorder=0, lw=0.8)
                if ind == 0:
                    ax[vals[ind]].text(
                        0.05,
                        0.95,
                        tracer + " ",
                        transform=ax[vals[ind]].transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color=c,
                    )

    fig.savefig(figname, bbox_inches="tight", dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    sigma = {"sym_correct": [5.0, 2.0, 2.0], "sym_wrong": [7.0, 3.0, 3.0]}
    sigma_prior_factor = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
    # First load up mock mean and add it to the fitting list.
    dataset_pk = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
    )

    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
    )

    for s, sig in enumerate(["sym_correct", "sym_wrong"]):
        for i, factor in enumerate(sigma_prior_factor):

            model = PowerBeutler2017(
                recon=dataset_pk.recon,
                isotropic=dataset_pk.isotropic,
                fix_params=["om"],
                marg="full",
                poly_poles=dataset_pk.fit_poles,
                correction=Correction.HARTLAP,
                broadband_type="spline",
            )
            model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
            model.set_default("beta", 0.4, min=0.1, max=0.7)
            model.set_default("sigma_nl_par", sigma[sig][0], min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma[sig][1], min=0.0, max=20.0, sigma=1.0 * factor, prior="gaussian")
            model.set_default("sigma_s", sigma[sig][2], min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")

            # Load in a pre-existing BAO template
            pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
            model.kvals, model.pksmooth, model.pkratio = pktemplate.T

            name = dataset_pk.name + f" mock mean {s} prior=" + str(i)
            fitter.add_model_and_dataset(model, dataset_pk, name=name)
            allnames.append(name)

            model = CorrBeutler2017(
                recon=dataset_xi.recon,
                isotropic=dataset_xi.isotropic,
                marg="full",
                fix_params=["om"],
                poly_poles=dataset_xi.fit_poles,
                correction=Correction.HARTLAP,
            )
            model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
            model.set_default("beta", 0.4, min=0.1, max=0.7)
            model.set_default("sigma_nl_par", sigma[sig][0], min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma[sig][1], min=0.0, max=20.0, sigma=1.0 * factor, prior="gaussian")
            model.set_default("sigma_s", sigma[sig][2], min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")

            # Load in a pre-existing BAO template
            pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
            model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

            name = dataset_xi.name + f" mock mean {s} prior=" + str(i)
            fitter.add_model_and_dataset(model, dataset_xi, name=name)
            allnames.append(name)

    # Submit all the jobs to NERSC. We have quite a few (156), so we'll
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
        stats = [[[] for _ in range(2)] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number, data bin sigma bin and prior bin
            data_bin = 0 if "Xi" in extra["name"] else 1
            sigma_bin = int(extra["name"].split("mock mean ")[1].split(" ")[0])
            prior_bin = int(extra["name"].split("prior=")[1].split(" ")[0])
            print(extra["name"], data_bin, sigma_bin, prior_bin)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            df["$\\alpha_{ap}$"] = (1.0 + df["$\\epsilon$"].to_numpy()) ** 3
            newweight = np.where(
                np.logical_and(
                    np.logical_and(df["$\\alpha_\\parallel$"] >= 0.8, df["$\\alpha_\\parallel$"] <= 1.2),
                    np.logical_and(df["$\\alpha_\\perp$"] >= 0.8, df["$\\alpha_\\perp$"] <= 1.2),
                ),
                weight,
                0.0,
            )

            # Compute some summary statistics and add them to a dictionary
            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha$",
                        "$\\alpha_{ap}$",
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                    ]
                ],
                newweight,
                axis=0,
            )

            stats[data_bin][sigma_bin].append(
                [
                    100.0 * (mean[0] - 1.0),
                    100.0 * (mean[1] - 1.0),
                    100.0 * np.sqrt(cov[0, 0]),
                    100.0 * np.sqrt(cov[1, 1]),
                ]
            )

        # Plot the error on the alpha parameters as a function of the width of the sigma prior
        plot_errors(sigma_prior_factor, stats, "/".join(pfn.split("/")[:-1]) + "/sigmaprior.png")
