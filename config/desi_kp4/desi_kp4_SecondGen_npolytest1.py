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
import pickle
from chainconsumer import ChainConsumer


def plot_errors(stats, data_sig, figname):

    covs = np.cov(stats, rowvar=False)

    labels = [r"$\sigma_{\alpha}$", r"$\sigma_{\alpha_{ap}}$", r"$\chi^{2}$"]
    colors = ["r", "b", "g"]
    fig, axes = plt.subplots(figsize=(7, 2), nrows=1, ncols=3, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.3)
    for i, (ax, vals, avgs, stds, l, c) in enumerate(
        zip(
            axes.T,
            np.array(stats[1:]).T[[4, 5, 10]],
            np.array(stats[0]).T[[4, 5, 10]],
            [np.sqrt(covs[0, 0]), np.sqrt(covs[1, 1]), 0.0],
            labels,
            colors,
        )
    ):

        ax[0].hist(vals, 10, color=c, histtype="stepfilled", alpha=0.2, density=False, zorder=0)
        ax[0].hist(vals, 10, color=c, histtype="step", alpha=1.0, lw=1.3, density=False, zorder=1)
        ax[0].axvline(data_sig[i], color="k", ls="-", zorder=2)
        if l != r"$\chi^{2}$":
            ax[0].axvline(avgs, color="k", ls="--", zorder=2)
            ax[0].axvline(stds, color="k", ls=":", zorder=2)
        ax[0].set_xlabel(l)

    axes[0, 0].set_ylabel(r"$N_{\mathrm{mocks}}$")

    fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)


# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/v2/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    tracers = {"LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]], "ELG_LOP": [[0.8, 1.1], [1.1, 1.6]], "QSO": [[0.8, 2.1]]}
    nmocks = {"LRG": [0, 25], "ELG_LOP": [0, 25], "QSO": [0, 25]}
    reconsmooth = {"LRG": 10, "ELG_LOP": 10, "QSO": 20}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [9.0, 6.0],
        ],
        "ELG_LOP": [[8.5, 6.0], [8.5, 6.0]],
        "QSO": [[9.0, 6.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [4.5, 3.0],
            [4.5, 3.0],
            [4.5, 3.0],
        ],
        "ELG_LOP": [[4.5, 3.0], [4.5, 3.0]],
        "QSO": [[3.5, 3.0]],
    }
    sigma_s = {"LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], "ELG_LOP": [[2.0, 2.0], [2.0, 2.0]], "QSO": [[2.0, 2.0]]}

    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = (
        "default_FKP"
        # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    )
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"

    plotnames = [f"{t}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
    datanames = [f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]

    allnames = []
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for r, recon in enumerate([None, "sym"]):

                model = CorrBeutler2017(
                    recon=recon,
                    isotropic=False,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=[0, 2],
                    correction=Correction.NONE,
                    n_poly=[-2, -1, 0],
                )
                model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
                dataset = CorrelationFunction_DESI_KP4(
                    recon=model.recon,
                    fit_poles=model.poly_poles,
                    min_dist=50.0,
                    max_dist=150.0,
                    realisation=None,
                    reduce_cov_factor=1,
                    datafile=name,
                )

                name = dataset.name + f" mock mean"
                fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i + 1])
                allnames.append(name)

                for j in range(len(dataset.mock_data)):
                    dataset.set_realisation(j)
                    name = dataset.name + f" realisation {j}"
                    fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i + 1])
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

        # Loop over all the fitters
        c = [ChainConsumer() for i in range(2 * len(datanames))]
        stats = [[[], []] for _ in range(len(datanames))]

        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            data_bin = datanames.index(extra["name"].split(" ")[3].lower())
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            stats_bin = recon_bin * len(datanames) + data_bin
            realisation = str(extra["name"].split()[-1]) if "realisation" in extra["name"] else "mean"
            print(extra["name"], data_bin, recon_bin, stats_bin, realisation)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            df["$\\alpha_{ap}$"] = (1.0 + df["$\\epsilon$"].to_numpy()) ** 3

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior.argmax()
            params = df.loc[max_post]
            params_dict = model.get_param_dict(chain[max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)

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
                weight,
                axis=0,
            )

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if realisation == "mean":
                extra.pop("color", None)
                c[stats_bin].add_chain(df, weights=weight, color="k", **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)
                figname = None
                mean_mean, cov_mean = mean, cov
            else:
                c[stats_bin].add_marker(params, **extra)
                # Get some useful properties of the fit, and plot the MAP model against the data if the bestfit alpha or alpha_ap are outliers compared to the mean fit
                diff = np.c_[params["$\\alpha_\\parallel$"], params["$\\alpha_\\perp$"]] - mean_mean[2:]
                outlier = diff @ np.linalg.inv(cov_mean[2:, 2:]) @ diff.T
                print(outlier, sp.stats.chi2.ppf(0.9545, 2, loc=0, scale=1))
                figname = (
                    "/".join(pfn.split("/")[:-1]) + "/" + extra["name"].replace(" ", "_") + "_bestfit.png"
                    if outlier > sp.stats.chi2.ppf(0.9545, 2, loc=0, scale=1)
                    else None
                )

            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(
                params_dict, display=False, figname=figname, title=extra["name"], c=colors[data_bin + 1]
            )

            stats[data_bin][recon_bin].append(
                [
                    mean[0],
                    mean[1],
                    mean[2],
                    mean[3],
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    np.sqrt(cov[3, 3]),
                    cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                    cov[2, 3] / np.sqrt(cov[2, 2] * cov[3, 3]),
                    new_chi_squared,
                ]
            )

        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    dataname = f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}"
                    data_bin = datanames.index(dataname.lower())
                    stats_bin = recon_bin * len(datanames) + data_bin

                    mean = np.mean(stats[data_bin][recon_bin][1:], axis=0)
                    cov = np.cov(stats[data_bin][recon_bin][1:], rowvar=False)

                    c[stats_bin].add_covariance(
                        mean[:4],
                        cov[:4, :4],
                        parameters=["$\\alpha$", "$\\alpha_{ap}$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                        color=colors[data_bin + 1],
                        plot_contour=True,
                        plot_point=False,
                        show_as_1d_prior=False,
                    )

                    truth = {
                        "$\\alpha$": 1.0,
                        "$\\alpha_{ap}$": 1.0,
                        "$\\alpha_\\perp$": 1.0,
                        "$\\alpha_\\parallel$": 1.0,
                        "$\\Sigma_{nl,||}$": sigma_nl_par[t][i][recon_bin],
                        "$\\Sigma_{nl,\\perp}$": sigma_nl_perp[t][i][recon_bin],
                        "$\\Sigma_s$": sigma_s[t][i][recon_bin],
                    }

                    plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha_\\parallel$",
                            "$\\alpha_\\perp$",
                        ],
                        legend=False,
                    )
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour2.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha$",
                            "$\\alpha_{ap}$",
                        ],
                        legend=False,
                    )

        # Plot histograms of the chi squared values and uncertainties for comparison to the data
        data_sigmas_prerecon = {
            "LRG": [
                [0.01914048897114662, 0.07010153295955529, 3.14631537e01],
                [0.024504953045061173, 0.12150022384792725, 4.49891758e01],
                [0.011767540688530032, 0.04823989601113721, 4.07940393e01],
            ],
            "ELG_LOP": [
                [0.09043183288774481, 0.2896776382884302, 53.8712616],
                [0.015996649062431367, 0.055653482135353816, 3.49656306e01],
            ],
            "QSO": [[0.029377613839533523, 0.1342904227952228, 31.78156724]],
        }
        data_sigmas_postrecon = {
            "LRG": [
                [9.7586959e-03, 3.1017183e-02, 3.8796406e01],
                [0.00962757040797696, 0.03070362805708815, 3.86308320e01],
                [0.007935953770661752, 0.029989033380867336, 2.97028557e01],
            ],
            "ELG_LOP": [[0.10079159, 0.36624245, 43.09215521], [0.011033510858277806, 0.03830975662578129, 5.17281058e01]],
            "QSO": [[0.05227388284071255, 0.05267882411109576, 51.08261214]],
        }
        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    dataname = f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}"
                    data_bin = datanames.index(dataname.lower())
                    stats_bin = recon_bin * len(datanames) + data_bin
                    data_sig = data_sigmas_prerecon[t][i] if recon_bin == 0 else data_sigmas_postrecon[t][i]

                    plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                    plot_errors(stats[data_bin][recon_bin], data_sig, "/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_errors.png")
