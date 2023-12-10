import os
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
import matplotlib.colors as mplc
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
    pfn, dir_name, file = setup(__file__, "/v3/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    colors = [mplc.cnames[color] for color in ["orange", "orangered", "firebrick", "lightskyblue", "steelblue", "seagreen", "black"]]

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

    count = 0
    allnames = []
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for r, recon in enumerate([None, "sym"]):

                model = CorrBeutler2017(
                    recon=recon,
                    isotropic=True,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=[0],
                    correction=Correction.NONE,
                    broadband_type="spline",
                    n_poly=[0, 2],
                )
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                model.set_default(
                    "sigma_nl",
                    np.sqrt((sigma_nl_par[t][i][r] ** 2 + 2.0 * sigma_nl_perp[t][i][r] ** 2) / 3.0),
                    min=0.0,
                    max=20.0,
                    sigma=2.0,
                    prior="gaussian",
                )
                model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
                dataset = CorrelationFunction_DESI_KP4(
                    recon=model.recon,
                    isotropic=model.isotropic,
                    fit_poles=model.poly_poles,
                    min_dist=50.0,
                    max_dist=150.0,
                    realisation=None,
                    reduce_cov_factor=1,
                    datafile=name,
                )

                name = dataset.name + f" mock mean"
                fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
                allnames.append(name)

                for j in range(len(dataset.mock_data)):
                    dataset.set_realisation(j)
                    name = dataset.name + f" realisation {j}"
                    fitter.add_model_and_dataset(model, dataset, name=name, color=colors[count])
                    allnames.append(name)
            count += 1

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
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

        for dataname in datanames:
            for recon in ["prerecon", "postrecon"]:
                plotname = f"{dataname}_{recon}"
                dir_name = "/".join(pfn.split("/")[:-1]) + "/" + plotname
                try:
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)
                except Exception:
                    pass

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

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior[weight > 0].argmax()
            params = df[weight > 0].iloc[max_post]
            params_dict = model.get_param_dict(chain[weight > 0][max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)
                # Compute some summary statistics and add them to a dictionary
                mean, cov = weighted_avg_and_cov(
                    df[
                        [
                            "$\\alpha$",
                            "$\\Sigma_{nl}$",
                            "$\\Sigma_s$",
                        ]
                    ],
                    weight,
                    axis=0,
                )
                print(mean, cov)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if realisation == "mean":
                extra.pop("color", None)
                c[stats_bin].add_chain(df, weights=weight, color="k", **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)
                figname = None
                mean_mean, cov_mean = mean, cov
            else:
                c[stats_bin].add_marker(params, **extra)
                dataname = extra["name"].split(" ")[3].lower()
                plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                figname = "/".join(pfn.split("/")[:-1]) + "/" + plotname + "/" + extra["name"].replace(" ", "_") + "_contour.png"
                if not os.path.isfile(figname):
                    extra.pop("color", None)
                    cc = ChainConsumer()
                    cc.add_chain(df, weights=weight, **extra, color=colors[data_bin])
                    cc.add_marker(df.iloc[max_post], **extra)
                    cc.plotter.plot(filename=figname)
                    figname = "/".join(pfn.split("/")[:-1]) + "/" + plotname + "/" + extra["name"].replace(" ", "_") + "_bestfit.png"
                else:
                    figname = None

            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(
                params_dict, display=False, figname=figname, title=extra["name"], c=colors[data_bin]
            )
            if realisation == "mean":
                print(25.0 * new_chi_squared, dof)

            if data_bin == 0 and (realisation == 2 or realisation == 21):
                df["weight"] = weight
                df.to_csv("/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_BOSSpoly.dat", index=False, sep=" ")

            stats[data_bin][recon_bin].append(
                [
                    mean[0],
                    mean[1],
                    mean[2],
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    new_chi_squared,
                    params_dict["alpha"],
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
                        mean[:3],
                        cov[:3, :3],
                        parameters=[
                            "$\\alpha$",
                            "$\\Sigma_{nl}$",
                            "$\\Sigma_s$",
                        ],
                        color=colors[data_bin],
                        plot_contour=True,
                        plot_point=False,
                        show_as_1d_prior=False,
                    )

                    truth = {
                        "$\\alpha$": 1.0,
                        "$\\Sigma_{nl}$": np.sqrt((sigma_nl_par[t][i][recon_bin] ** 2 + 2.0 * sigma_nl_perp[t][i][recon_bin] ** 2) / 3.0),
                        "$\\Sigma_s$": sigma_s[t][i][recon_bin],
                    }

                    plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha$",
                            "$\\Sigma_{nl}$",
                            "$\\Sigma_s$",
                        ],
                        legend=False,
                    )

                    np.save("/".join(pfn.split("/")[:-1]) + "/Summary_" + plotname + f".npy", stats[data_bin][recon_bin])
