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
def plot_errors(stats, figname):

    fig, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=2, squeeze=True)
    plt.subplots_adjust(left=0.15, top=0.97, bottom=0.17, right=0.8, hspace=0.40, wspace=0.10)

    ax1 = fig.add_subplot(axes[0, 0])
    ax2 = fig.add_subplot(axes[0, 1])
    ax3 = fig.add_subplot(axes[1, 0])
    ax4 = fig.add_subplot(axes[1, 1])

    print(100.0 * stats[2, :, 1], 100.0 * stats[2, :, 2])
    print(100.0 * stats[2, :, 5], 100.0 * stats[2, :, 6])

    for j in range(3):
        bias1, bias2 = 100.0 * stats[j, :, 1], 100.0 * stats[j, :, 2]
        err1, err2 = 100.0 * stats[j, :, 5], 100.0 * stats[j, :, 6]

        ax1.plot(np.linspace(-0.5, 0.5, 500), gaussian_kde(bias1)(np.linspace(-0.5, 0.5, 500)))
        ax3.plot(np.linspace(-1.0, 1.0, 500), gaussian_kde(bias1)(np.linspace(-1.0, 1.0, 500)))
        ax2.plot(np.linspace(0.6, 0.7, 500), gaussian_kde(err1)(np.linspace(0.6, 0.9, 500)))
        ax4.plot(np.linspace(2.0, 2.5, 500), gaussian_kde(err2)(np.linspace(2.0, 2.5, 500)))

    ax1.set_xlabel(r"$\Delta\alpha_{\mathrm{iso}}\,(\%)$")
    ax3.set_xlabel(r"$\Delta\alpha_{\mathrm{ap}}\,(\%)$")
    ax2.set_xlabel(r"$\sigma_{\alpha_{\mathrm{iso}}}\,(\%)$")
    ax4.set_xlabel(r"$\sigma_{\alpha_{\mathrm{ap}}}\,(\%)$")
    ax1.set_ylabel(r"$N_{\mathrm{mocks}}$")
    ax2.set_yticklabels([])
    ax3.set_ylabel(r"$N_{\mathrm{mocks}}$")
    ax4.set_yticklabels([])

    ax1.axvline(0.0, color="k", ls="-", zorder=0, lw=0.8)
    ax3.axvline(0.0, color="k", ls="-", zorder=0, lw=0.8)

    fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    sigma = {None: [9.5, 5.0, 2.0], "sym": [5.0, 2.0, 2.0]}
    prior = ["fixed", "gaussian", "flat"]

    colors = ["#ABD465", "#1C8275", "#232C3B"]

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
        reduce_cov_factor=1.0 / 16.0,
        datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
    )

    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1.0 / 16.0,
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
    )

    # Loop over pre- and post-recon measurements
    for i, p in enumerate(prior):

        model = PowerBeutler2017(
            recon=dataset_pk.recon,
            isotropic=dataset_pk.isotropic,
            fix_params=["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"] if p == "fixed" else ["om"],
            marg="full",
            poly_poles=dataset_pk.fit_poles,
            correction=Correction.HARTLAP,
        )
        model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
        model.set_default("beta", 0.4, min=0.1, max=0.7)
        if p == "fixed":
            model.set_default("sigma_nl_par", sigma["sym"][0])
            model.set_default("sigma_nl_perp", sigma["sym"][1])
            model.set_default("sigma_s", sigma["sym"][2])
        if p == "gaussian":
            model.set_default("sigma_nl_par", sigma["sym"][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma["sym"][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
            model.set_default("sigma_s", sigma["sym"][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        if p == "flat":
            model.set_default("sigma_nl_par", sigma["sym"][0], min=0.0, max=20.0)
            model.set_default("sigma_nl_perp", sigma["sym"][1], min=0.0, max=20.0)
            model.set_default("sigma_s", sigma["sym"][2], min=0.0, max=20.0)

        # Load in a pre-existing BAO template
        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
        model.kvals, model.pksmooth, model.pkratio = pktemplate.T

        name = dataset_pk.name + f" mock mean sigma prior = " + str(p)
        fitter.add_model_and_dataset(model, dataset_pk, name=name, color=colors[i])
        allnames.append(name)

        for j in range(len(dataset_pk.mock_data)):
            dataset_pk.set_realisation(j)
            name = dataset_pk.name + f" realisation {j} sigma prior = " + str(p)
            fitter.add_model_and_dataset(model, dataset_pk, name=name, color=colors[i])
            allnames.append(name)

        model = CorrBeutler2017(
            recon=dataset_xi.recon,
            isotropic=dataset_xi.isotropic,
            fix_params=["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"] if p == "fixed" else ["om"],
            marg="full",
            poly_poles=dataset_xi.fit_poles,
            correction=Correction.HARTLAP,
        )
        model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
        model.set_default("beta", 0.4, min=0.1, max=0.7)
        if p == "fixed":
            model.set_default("sigma_nl_par", sigma["sym"][0])
            model.set_default("sigma_nl_perp", sigma["sym"][1])
            model.set_default("sigma_s", sigma["sym"][2])
        if p == "gaussian":
            model.set_default("sigma_nl_par", sigma["sym"][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma["sym"][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
            model.set_default("sigma_s", sigma["sym"][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        if p == "flat":
            model.set_default("sigma_nl_par", sigma["sym"][0], min=0.0, max=20.0)
            model.set_default("sigma_nl_perp", sigma["sym"][1], min=0.0, max=20.0)
            model.set_default("sigma_s", sigma["sym"][2], min=0.0, max=20.0)

        # Load in a pre-existing BAO template
        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

        name = dataset_xi.name + f" mock mean sigma prior = " + str(p)
        fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[i])
        allnames.append(name)

        for j in range(len(dataset_xi.mock_data)):
            dataset_xi.set_realisation(j)
            name = dataset_xi.name + f" realisation {j} sigma prior = " + str(p)
            fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[i])
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
        stats = [[[] for _ in range(len(prior))] for _ in range(len(datanames))]
        all_samples = [[[] for _ in range(len(prior))] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            if "Prerecon" in extra["name"]:
                continue

            # Get the realisation number and redshift bin
            data_bin = 0 if "Xi" in extra["name"] else 1
            sigma_bin = prior.index(extra["name"].split("sigma prior = ")[1])
            realisation = int(extra["name"].split("realisation ")[1].split(" ")[0]) if "realisation" in extra["name"] else -1
            print(data_bin, sigma_bin, realisation)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())
            print(df.keys())

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

            if realisation == 1:
                if sigma_bin == 0:
                    all_samples[data_bin][sigma_bin].extend(
                        np.c_[
                            df["$\\alpha$"].to_numpy(),
                            df["$\\alpha_{ap}$"].to_numpy(),
                            df["$\\alpha_\\parallel$"].to_numpy(),
                            df["$\\alpha_\\perp$"].to_numpy(),
                            df["$\\beta$"].to_numpy(),
                            newweight,
                            np.ones(len(newweight)) * evidence,
                        ]
                    )
                else:
                    all_samples[data_bin][sigma_bin].extend(
                        np.c_[
                            df["$\\alpha$"].to_numpy(),
                            df["$\\alpha_{ap}$"].to_numpy(),
                            df["$\\alpha_\\parallel$"].to_numpy(),
                            df["$\\alpha_\\perp$"].to_numpy(),
                            df["$\\beta$"].to_numpy(),
                            df["$\\Sigma_{nl,||}$"].to_numpy(),
                            df["$\\Sigma_{nl,\\perp}$"].to_numpy(),
                            df["$\\Sigma_s$"].to_numpy(),
                            newweight,
                            np.ones(len(newweight)) * evidence,
                        ]
                    )

            stats[data_bin][sigma_bin].append(
                [
                    realisation,
                    mean[0] - 1.0,
                    mean[1] - 1.0,
                    mean[2] - 1.0,
                    mean[3] - 1.0,
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    np.sqrt(cov[3, 3]),
                    cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                    cov[2, 3] / np.sqrt(cov[2, 2] * cov[3, 3]),
                ]
            )

        truth = {
            "$\\alpha$": 1.0,
            "$\\alpha_{ap}$": 1.0,
            "$\\alpha_\\perp$": 1.0,
            "$\\alpha_\\parallel$": 1.0,
            "$\\Sigma_{nl,||}$": sigma["sym"][0],
            "$\\Sigma_{nl,\\perp}$": sigma["sym"][1],
            "$\\Sigma_{s}$": sigma["sym"][2],
        }

        for i, dname in enumerate(datanames):
            c = ChainConsumer()
            for j, pname in enumerate(prior):
                samples = np.array(all_samples[i][j])
                print(i, j, np.amin(samples[:, -1]))

                if j == 0:
                    params = [
                        "$\\alpha$",
                        "$\\alpha_{ap}$",
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                        "$\\beta$",
                    ]
                else:
                    params = [
                        "$\\alpha$",
                        "$\\alpha_{ap}$",
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                        "$\\beta$",
                        "$\\Sigma_{nl,||}$",
                        "$\\Sigma_{nl,\\perp}$",
                        "$\\Sigma_{s}$",
                    ]
                c.add_chain(
                    samples[:, :-2],
                    weights=samples[:, -2] * np.exp((samples[:, -1] - np.amax(samples[:, -1]))),
                    name=pname,
                    parameters=params,
                )

            c.plotter.plot(
                filename=["/".join(pfn.split("/")[:-1]) + "/" + dname + f"_contour.png"],
                truth=truth,
                parameters=[
                    "$\\alpha$",
                    "$\\alpha_{ap}$",
                    "$\\alpha_\\parallel$",
                    "$\\alpha_\\perp$",
                    "$\\beta$",
                    "$\\Sigma_{nl,||}$",
                    "$\\Sigma_{nl,\\perp}$",
                    "$\\Sigma_{s}$",
                ],
                legend=False,
            )

            # Plot the bias and error on the alpha parameters as a function of the choice of the sigma prior
            plot_errors(np.array(stats[i]), "/".join(pfn.split("/")[:-1]) + "/" + dname + "_alphas.png")
