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
import matplotlib.gridspec as gridspec


# Convenience function to plot histograms of the errors and cross-correlation coefficients
def plot_errors(sigma_prior_factor, stats, figname):

    mosaic = """AB
                CD
                EF
                GH"""
    fig = plt.figure(layout="constrained")
    left, right = fig.subfigures(nrows=1, ncols=2, width_ratios=[1.245, 1])
    axxi = left.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "bottom": 0.1,
            "top": 0.95,
            "left": 0.1,
            "right": 0.95,
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )
    axpk = right.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "bottom": 0.1,
            "top": 0.95,
            "left": 0.1,
            "right": 0.95,
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )

    for data_bin, ax in enumerate([axxi, axpk]):
        c = "#ff7f0e" if data_bin == 0 else "#1f77b4"
        tracer = r"$\xi(s)$" if data_bin == 0 else r"$P(k)$"

        for sigma_bin, vals in enumerate([["A", "C", "E", "G"], ["B", "D", "F", "H"]]):
            sig = r"$Fiducial\,\Sigma$" if sigma_bin == 0 else r"$Incorrect\,\Sigma$"
            statsmean = np.mean(np.array(stats[data_bin][sigma_bin]), axis=0)
            statsstd = np.std(np.array(stats[data_bin][sigma_bin]), axis=0) / 5.0
            for ind, (label, range) in enumerate(
                zip(
                    [
                        r"$\Delta \alpha_{\mathrm{iso}}\,(\%)$",
                        r"$\Delta \alpha_{\mathrm{ap}}\,(\%)$",
                        r"$\sigma_{\alpha_{\mathrm{iso}}}\,(\%)$",
                        r"$\sigma_{\alpha_{\mathrm{ap}}}\,(\%)$",
                    ],
                    [[-0.35, 0.35], [-0.95, 0.95], [0.285, 0.33], [0.90, 1.15]],
                )
            ):
                ax[vals[ind]].plot(sigma_prior_factor, statsmean[:, ind], color=c, zorder=1, alpha=0.75, lw=0.8)
                ax[vals[ind]].fill_between(
                    sigma_prior_factor,
                    statsmean[:, ind] - statsstd[:, ind],
                    statsmean[:, ind] + statsstd[:, ind],
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
                        0.12,
                        0.95,
                        tracer + " " + sig,
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

    tracers = {
        "LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
        "ELG_LOP": [[0.8, 1.1], [1.1, 1.6]],
        "QSO": [[0.8, 2.1]],
        "BGS_BRIGHT-21.5": [[0.1, 0.4]],
    }
    reconsmooth = {"LRG": 10, "ELG_LOP": 10, "QSO": 30, "BGS_BRIGHT-21.5": 15}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [9.0, 6.0],
        ],
        "ELG_LOP": [[8.5, 6.0], [8.5, 6.0]],
        "QSO": [[9.0, 6.0]],
        "BGS_BRIGHT-21.5": [[10.0, 8.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [4.5, 3.0],
            [4.5, 3.0],
            [4.5, 3.0],
        ],
        "ELG_LOP": [[4.5, 3.0], [4.5, 3.0]],
        "QSO": [[3.5, 3.0]],
        "BGS_BRIGHT-21.5": [[6.5, 3.0]],
    }
    sigma_s = {
        "LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        "ELG_LOP": [[2.0, 2.0], [2.0, 2.0]],
        "QSO": [[2.0, 2.0]],
        "BGS_BRIGHT-21.5": [[2.0, 2.0]],
    }
    sigma_prior_factor = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Loop over the mocktypes
    allnames = []

    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = (
        "default_FKP"
        # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    )
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"

    # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
    # First load up mock mean and add it to the fitting list.
    t = "ELG_LOP"
    zs = tracers[t][1]
    name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_pk.pkl"
    dataset_pk = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1,
        datafile=name,
    )

    name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1,
        datafile=name,
    )

    for s, sig in enumerate([0.0, 1.0]):
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
            model.set_default("sigma_nl_par", sigma_nl_par[t][1][0] + 2.0 * sig, min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma_nl_perp[t][1][0] + sig, min=0.0, max=20.0, sigma=1.0 * factor, prior="gaussian")
            model.set_default("sigma_s", sigma_s[t][1][0] + 2.0 * sig, min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")

            for j in range(len(dataset_pk.mock_data)):
                dataset_pk.set_realisation(j)
                name = dataset_pk.name + f" realisation {j} {s} prior=" + str(i)
                fitter.add_model_and_dataset(model, dataset_pk, name=name)

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
            model.set_default("sigma_nl_par", sigma_nl_par[t][1][0] + 2.0 * sig, min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma_nl_perp[t][1][0] + sig, min=0.0, max=20.0, sigma=1.0 * factor, prior="gaussian")
            model.set_default("sigma_s", sigma_s[t][1][0] + 2.0 * sig, min=0.0, max=20.0, sigma=2.0 * factor, prior="gaussian")

            for j in range(len(dataset_xi.mock_data)):
                dataset_xi.set_realisation(j)
                name = dataset_xi.name + f" realisation {j} {s} prior=" + str(i)
                fitter.add_model_and_dataset(model, dataset_xi, name=name)

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
        stats = [[[[] for _ in range(len(dataset_xi.mock_data))] for _ in range(2)] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number, data bin sigma bin and prior bin
            data_bin = 0 if "Xi" in extra["name"] else 1
            realisation = int(extra["name"].split("realisation ")[1].split(" ")[0])
            sigma_bin = int(extra["name"].split("realisation ")[1].split(" ")[1])
            prior_bin = int(extra["name"].split("prior=")[1].split(" ")[0])
            print(extra["name"], data_bin, sigma_bin, realisation, prior_bin)

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

            stats[data_bin][sigma_bin][realisation].append(
                [
                    100.0 * (mean[0] - 1.0),
                    100.0 * (mean[1] - 1.0),
                    100.0 * np.sqrt(cov[0, 0]),
                    100.0 * np.sqrt(cov[1, 1]),
                ]
            )

        # Plot the error on the alpha parameters as a function of the width of the sigma prior
        plot_errors(sigma_prior_factor, stats, "/".join(pfn.split("/")[:-1]) + "/sigmaprior.png")
