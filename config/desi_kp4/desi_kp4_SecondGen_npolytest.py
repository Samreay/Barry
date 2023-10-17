import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import Optimiser
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
    sampler = Optimiser(temp_dir=dir_name)

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    tracers = {"LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]], "ELG_LOP": [[0.8, 1.1], [1.1, 1.6]], "QSO": [[0.8, 2.1]]}
    nmocks = {"LRG": [0, 25], "ELG_LOP": [0, 25], "QSO": [0, 25]}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [8.0, 5.5],
        ],
        "ELG_LOP": [[9.0, 6.0], [9.0, 6.0]],
        "QSO": [[11.0, 0.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [3.5, 3.0],
            [4.0, 2.0],
            [5.0, 3.5],
        ],
        "ELG_LOP": [[4.5, 4.0], [4.5, 4.0]],
        "QSO": [[0.0, 0.0]],
    }
    sigma_s = {"LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], "ELG_LOP": [[7.0, 10.0], [7.0, 0.0]], "QSO": [[2.0, 0.0]]}

    allnames = []
    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = "default_FKP"  # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            rec = [None] if t == "QSO" else [None, "sym"]
            for r, recon in enumerate(rec):
                for n, n_poly in enumerate([[], [-2, -1, 0], [0, 2], [-2, 0, 2]]):

                    model = CorrBeutler2017(
                        recon=recon,
                        isotropic=False,
                        marg="full",
                        fix_params=["om"],
                        poly_poles=[0, 2],
                        correction=Correction.NONE,
                        n_poly=n_poly,
                    )
                    model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                    # Load in a pre-existing BAO template
                    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                    name = f"DESI_SecondGen_sm10_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
                    dataset = CorrelationFunction_DESI_KP4(
                        recon=model.recon,
                        fit_poles=model.poly_poles,
                        min_dist=50.0,
                        max_dist=150.0,
                        realisation=None,
                        reduce_cov_factor=1,
                        datafile=name,
                    )

                    name = dataset.name + f" mock mean n_poly=" + str(n)
                    fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i - 1])
                    allnames.append(name)

                    for j in range(len(dataset.mock_data)):
                        dataset.set_realisation(j)
                        name = dataset.name + f" realisation {j} n_poly=" + str(n)
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

        # Set up a ChainConsumer instance. Plot the MAP for individual realisations and a contour for the mock average
        data_names = [r"$\xi_{s}$", r"$P(k)$"]
        c = ChainConsumer()

        # Loop over all the fitters
        stats = [[[] for _ in range(len(data_names))] for _ in range(len(sampler_names))]
        output = {k: [] for k in sampler_names}
        for sampler_bin, (sampler, sampler_name) in enumerate(zip(samplers, sampler_names)):
            print(sampler_bin, sampler, sampler_name)
            fitter.set_sampler(sampler)

            for posterior, weight, chain, evidence, model, data, extra in fitter.load():

                # Get the realisation number and redshift bin
                data_bin = 0 if "Pre" in extra["name"] else 1

                # Store the chain in a dictionary with parameter names
                df = pd.DataFrame(chain, columns=model.get_labels())

                # Compute alpha_par and alpha_perp for each point in the chain
                alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
                df["$\\alpha_\\parallel$"] = alpha_par
                df["$\\alpha_\\perp$"] = alpha_perp
                mean, cov = weighted_avg_and_cov(
                    df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"]],
                    weight,
                    axis=0,
                )

                if "mock mean" in extra["name"]:
                    name = f"{sampler_name} + {data_names[data_bin]}"
                    c.add_chain(df, weights=weight, name=name, plot_contour=True, plot_point=False, show_as_1d_prior=False)

                # Use chainconsumer to get summary statistics for each chain
                cc = ChainConsumer()
                cc.add_chain(df, weights=weight)
                onesigma = cc.analysis.get_summary()
                cc.configure(summary_area=0.9545)
                twosigma = cc.analysis.get_summary()

                # if None in twosigma["$\\alpha_\\parallel$"] or None in twosigma["$\\alpha_\\perp$"]:
                #    continue

                # Store the summary statistics
                if extra["realisation"] is not None:
                    stats[sampler_bin][data_bin].append(
                        [
                            extra["realisation"],
                            onesigma["$\\alpha_\\parallel$"][1],
                            onesigma["$\\alpha_\\perp$"][1],
                            onesigma["$\\alpha_\\parallel$"][2] - onesigma["$\\alpha_\\parallel$"][0],
                            onesigma["$\\alpha_\\perp$"][2] - onesigma["$\\alpha_\\perp$"][0],
                            twosigma["$\\alpha_\\parallel$"][2] - twosigma["$\\alpha_\\parallel$"][0],
                            twosigma["$\\alpha_\\perp$"][2] - twosigma["$\\alpha_\\perp$"][0],
                        ]
                    )

        truth = {
            "$\\alpha_\\perp$": 1.0,
            "$\\alpha_\\parallel$": 1.0,
            "$\\Sigma_{nl,||}$": 5.1,
            "$\\Sigma_{nl,\\perp}$": 1.6,
            "$\\Sigma_s$": 0.0,
        }

        c.configure(bins=20, bar_shade=True)
        c.plotter.plot_summary(
            filename=["/".join(pfn.split("/")[:-1]) + "/summary.png"],
            truth=truth,
            parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\Sigma_s$"],
            extents=[(0.98, 1.02), (0.98, 1.02)],
        )

        # Plot the summary statistics
        print(stats)
        stats = np.array(stats)
        print(stats)

        fig, axes = plt.subplots(figsize=(4, 5), nrows=6, ncols=1, sharex=True, squeeze=False)
        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
        for panel in range(6):
            axes[panel, 0].axhline(0.0, color="k", ls="--", zorder=0, lw=0.8)

            for data_bin, label in enumerate(data_names):

                diff = stats[:, data_bin, :, panel + 1] - stats[0, data_bin, :, panel + 1]
                mean_diff, std_diff = np.mean(diff, axis=1), np.std(diff, axis=1)

                axes[panel, 0].errorbar(
                    np.arange(len(sampler_names[1:])) + (2 * data_bin - 1) * 0.1,
                    100.0 * mean_diff[1:],
                    yerr=100.0 * std_diff[1:] / np.sqrt(25.0),
                    marker="o",
                    ls="None",
                    label=label if panel == 3 else None,
                )

        axes[0, 0].set_ylabel("$\\Delta \\alpha_{\\perp} (\\%)$")
        axes[1, 0].set_ylabel("$\\Delta \\alpha_{||} (\\%)$")
        axes[2, 0].set_ylabel("$\\Delta \\sigma^{68\\%}_{\\alpha_{||}} (\\%)$")
        axes[3, 0].set_ylabel("$\\Delta \\sigma^{68\\%}_{\\alpha_{\\perp}} (\\%)$")
        axes[4, 0].set_ylabel("$\\Delta \\sigma^{95\\%}_{\\alpha_{||}} (\\%)$")
        axes[5, 0].set_ylabel("$\\Delta \\sigma^{95\\%}_{\\alpha_{\\perp}} (\\%)$")
        axes[0, 0].set_ylim(-0.07, 0.07)
        axes[1, 0].set_ylim(-0.06, 0.06)
        axes[2, 0].set_ylim(-0.015, 0.015)
        axes[3, 0].set_ylim(-0.015, 0.015)
        axes[4, 0].set_ylim(-0.04, 0.04)
        axes[5, 0].set_ylim(-0.018, 0.018)
        axes[3, 0].legend(
            loc="center right",
            bbox_to_anchor=(1.25, 1.0),
            frameon=False,
        )
        plt.setp(axes, xticks=[0, 1, 2, 3], xticklabels=sampler_names[1:])
        plt.xticks(rotation=30)
        fig.savefig("/".join(pfn.split("/")[:-1]) + "/bias.png", bbox_inches="tight", transparent=True, dpi=300)
