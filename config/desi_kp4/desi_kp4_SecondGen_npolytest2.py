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

# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

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
            [8.0, 5.5],
        ],
        "ELG_LOP": [[9.0, 6.0], [8.5, 5.0]],
        "QSO": [[11.0, 8.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [3.5, 3.0],
            [4.0, 2.0],
            [5.0, 3.5],
        ],
        "ELG_LOP": [[4.5, 4.0], [4.0, 4.0]],
        "QSO": [[2.0, 2.0]],
    }
    sigma_s = {"LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], "ELG_LOP": [[4.0, 6.0], [3.0, 2.0]], "QSO": [[2.0, 2.0]]}

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

    try:
        with open("/".join(pfn.split("/")[:-1]) + "/stats.pkl", "rb") as f:
            stats = pickle.load(f)
    except:

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
                        n_poly=[0, 2],
                    )
                    model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
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
                    fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i - 1])
                    allnames.append(name)

                    for j in range(len(dataset.mock_data)):
                        dataset.set_realisation(j)
                        name = dataset.name + f" realisation {j}"
                        fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i - 1])
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
            stats = [[[[], []] for _ in range(len(datanames))] for _ in range(4)]

            for posterior, weight, chain, evidence, model, data, extra in fitter.load():

                data_bin = datanames.index(extra["name"].split(" ")[3].lower())
                recon_bin = 0 if "Prerecon" in extra["name"] else 1
                poly_bin = int(extra["name"].split("n_poly=")[1].split(" ")[0])
                stats_bin = recon_bin * 4 + poly_bin
                print(extra["name"], data_bin, recon_bin, poly_bin, stats_bin)

                # Store the chain in a dictionary with parameter names
                df = pd.DataFrame(chain, columns=model.get_labels())

                model.set_data(data)
                params_dict = model.get_param_dict(chain[0])
                for name, val in params_dict.items():
                    model.set_default(name, val)

                new_chi_squared, dof, bband, mods, smooths = model.simple_plot(params_dict, display=False)

                # Compute alpha_par and alpha_perp for each point in the chain
                alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
                df["$\\alpha_\\parallel$"] = alpha_par
                df["$\\alpha_\\perp$"] = alpha_perp

                style = "o" if "mock mean" in extra["name"] else "."

                plotname = f"{plotnames[data_bin]}_prerecon" if recon_bin == 0 else f"{plotnames[data_bin]}_postrecon"

                # Store the summary statistics
                stats[poly_bin][data_bin][recon_bin].append(
                    [
                        extra["realisation"],
                        df["$\\alpha$"].to_numpy()[0],
                        df["$\\epsilon$"].to_numpy()[0],
                        df["$\\alpha_\\parallel$"].to_numpy()[0],
                        df["$\\alpha_\\perp$"].to_numpy()[0],
                        df["$\\beta$"].to_numpy()[0],
                        df["$\\Sigma_{nl,||}$"].to_numpy()[0],
                        df["$\\Sigma_{nl,\\perp}$"].to_numpy()[0],
                        df["$\\Sigma_s$"].to_numpy()[0],
                        new_chi_squared,
                        dof,
                    ]
                )

            with open("/".join(pfn.split("/")[:-1]) + "/stats.pkl", "wb") as f:
                pickle.dump(stats, f)

    if fitter.should_plot():

        """names = [
            "$\\alpha$",
            "$\\epsilon$",
            "$\\alpha_\\parallel$",
            "$\\alpha_\\perp$",
            "$\\beta$",
            "$\\Sigma_{nl,||}$",
            "$\\Sigma_{nl,\\perp}$",
            "$\\Sigma_s$",
        ]
        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    for poly_bin in range(4):
                        dataname = f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}"
                        data_bin = datanames.index(dataname.lower())

                        c = ChainConsumer()
                        print(stats[poly_bin][data_bin][recon_bin][0][1:-2])
                        c.add_marker(stats[poly_bin][data_bin][recon_bin][0][1:-2], parameters=names, marker_style="X")
                        for j in range(25):
                            print(stats[poly_bin][data_bin][recon_bin][j][1:-2])
                            c.add_marker(stats[poly_bin][data_bin][recon_bin][j][1:-2], parameters=names, marker_style=".")

                        truth = {
                            "$\\alpha_\\perp$": 1.0,
                            "$\\alpha_\\parallel$": 1.0,
                            "$\\Sigma_{nl,||}$": sigma_nl_par[t][i][recon_bin],
                            "$\\Sigma_{nl,\\perp}$": sigma_nl_perp[t][i][recon_bin],
                            "$\\Sigma_s$": sigma_s[t][i][recon_bin],
                        }

                        plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                        c.plotter.plot(
                            filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_npoly{poly_bin}_contour.png"],
                            truth=truth,
                            parameters=[
                                "$\\alpha_\\parallel$",
                                "$\\alpha_\\perp$",
                            ],
                            legend=False,
                        )"""

        # Summarise the many realisations in a different way
        stats2 = np.zeros((6, len(datanames), 2, 3))
        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    for poly_bin, poly in enumerate([0, 2, 3]):
                        dataname = f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}"
                        data_bin = datanames.index(dataname.lower())

                        x0 = np.array(stats[1][data_bin][recon_bin][1:])
                        x = np.array(stats[poly][data_bin][recon_bin][1:])
                        delta_apar, delta_aperp = (x - x0)[:, 3], (x - x0)[:, 4]
                        delta_std_apar, delta_std_aperp = (np.std(x, axis=0) - np.std(x0, axis=0))[3], (
                            np.std(x, axis=0) - np.std(x0, axis=0)
                        )[4]

                        stats2[0, data_bin, recon_bin, poly_bin] = np.mean(delta_apar, axis=0)
                        stats2[1, data_bin, recon_bin, poly_bin] = np.mean(delta_aperp, axis=0)
                        stats2[2, data_bin, recon_bin, poly_bin] = np.std(delta_apar, axis=0) / np.sqrt(25.0)
                        stats2[3, data_bin, recon_bin, poly_bin] = np.std(delta_aperp, axis=0) / np.sqrt(25.0)
                        stats2[4, data_bin, recon_bin, poly_bin] = delta_std_apar
                        stats2[5, data_bin, recon_bin, poly_bin] = delta_std_aperp

        print(stats2)

        # Plot the summary statistics
        fig, axes = plt.subplots(figsize=(8, 5), nrows=4, ncols=1, sharex=True, squeeze=False)
        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
        for panel in range(2):
            axes[panel, 0].axhline(0.0, color="k", ls="--", zorder=0, lw=0.8)

            for data_bin, label in enumerate(datanames):

                print(data_bin + np.arange(3) * 0.1 - 0.3, stats2[panel, data_bin, 0], stats2[panel + 2, data_bin, 0])
                print(data_bin + np.arange(3) * 0.1, stats2[panel, data_bin, 1], stats2[panel + 2, data_bin, 1])

                axes[panel, 0].errorbar(
                    data_bin + np.arange(3) * 0.1 - 0.25,
                    100.0 * stats2[panel, data_bin, 0],
                    yerr=100.0 * stats2[panel + 2, data_bin, 0],
                    marker="o",
                    ls="None",
                    label=label if panel == 1 else None,
                    c="r",
                )
                axes[panel, 0].errorbar(
                    data_bin + np.arange(3) * 0.1 + 0.05,
                    100.0 * stats2[panel, data_bin, 1],
                    yerr=100.0 * stats2[panel + 2, data_bin, 1],
                    marker="o",
                    ls="None",
                    label=label if panel == 1 else None,
                    c="b",
                )
                axes[panel + 2, 0].errorbar(
                    data_bin + np.arange(3) * 0.1 - 0.25,
                    100.0 * stats2[panel + 4, data_bin, 0],
                    marker="o",
                    ls="None",
                    label=label if panel == 1 else None,
                    c="r",
                )
                axes[panel + 2, 0].errorbar(
                    data_bin + np.arange(3) * 0.1 + 0.05,
                    100.0 * stats2[panel + 4, data_bin, 1],
                    marker="o",
                    ls="None",
                    label=label if panel == 1 else None,
                    c="b",
                )

        axes[0, 0].set_ylabel("$\\Delta \\alpha_{||} (\\%)$")
        axes[1, 0].set_ylabel("$\\Delta \\alpha_{\\perp} (\\%)$")
        axes[2, 0].set_ylabel("$\\Delta \\sigma^{68\\%}_{\\alpha_{||}} (\\%)$")
        axes[3, 0].set_ylabel("$\\Delta \\sigma^{68\\%}_{\\alpha_{\\perp}} (\\%)$")
        # axes[0, 0].set_ylim(-0.07, 0.07)
        # axes[1, 0].set_ylim(-0.06, 0.06)
        # axes[2, 0].set_ylim(-0.015, 0.015)
        # axes[3, 0].set_ylim(-0.015, 0.015)
        # axes[3, 0].legend(
        #    loc="center right",
        #    bbox_to_anchor=(1.25, 1.0),
        #    frameon=False,
        # )
        plt.setp(axes, xticks=[0, 1, 2, 3, 4, 5], xticklabels=datanames)
        plt.xticks(rotation=30)
        fig.savefig("/".join(pfn.split("/")[:-1]) + "/bias.png", bbox_inches="tight", transparent=True, dpi=300)
