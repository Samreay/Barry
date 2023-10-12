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
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.

# Convenience function to plot histograms of the errors and cross-correlation coefficients
def plot_alphas(data, figname, plotnames):

    colors = ["#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Split up Pk and Xi
    fig = plt.figure(figsize=(10, 3))
    axes = gridspec.GridSpec(2, 1, figure=fig, left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
    axes1 = axes[0, 0].subgridspec(1, np.shape(data)[0], hspace=0.0, wspace=0.0)
    axes2 = axes[1, 0].subgridspec(1, np.shape(data)[0], hspace=0.0, wspace=0.0)

    print(np.shape(data))

    # Further split up Xi into different polynomial types
    for index in range(np.shape(data)[0]):
        print(data[index], plotnames[index])
        ax1 = fig.add_subplot(axes1[index])
        ax2 = fig.add_subplot(axes2[index])

        xis = data[index]

        print(xis[:, 0], xis[:, 2] * 100.0)

        ax1.plot(xis[:, 0], xis[:, 1] * 100.0, color=colors[index], zorder=1, alpha=0.75, lw=0.8)
        ax2.plot(xis[:, 0], xis[:, 2] * 100.0, color=colors[index], zorder=1, alpha=0.75, lw=0.8)
        ax1.fill_between(
            xis[:, 0],
            (xis[:, 1] - xis[:, 3]) * 100.0,
            (xis[:, 1] + xis[:, 3]) * 100.0,
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax2.fill_between(
            xis[:, 0],
            (xis[:, 2] - xis[:, 4]) * 100.0,
            (xis[:, 2] + xis[:, 4]) * 100.0,
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        # ax1.set_xlim(1.3, 6.7)
        # ax2.set_xlim(1.3, 6.7)
        # ax1.set_ylim(-0.75, 0.75)
        # ax2.set_ylim(-0.35, 0.35)
        ax2.set_xlabel(r"$\Sigma_{nl,||}$")
        if n_poly == 0:
            ax1.set_ylabel(r"$\alpha_{||} - 1\,(\%)$")
            ax2.set_ylabel(r"$\alpha_{\perp} - 1\,(\%)$")
        else:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        ax1.set_xticklabels([])
        for val, ls in zip([-0.1, 0.0, 0.1], [":", "--", ":"]):
            ax1.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
            ax2.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
        ax1.axvline(4.75, color="k", ls=":", zorder=0, lw=0.8)
        ax2.axvline(4.75, color="k", ls=":", zorder=0, lw=0.8)
        ax1.text(
            0.05,
            0.95,
            f"{plotnames[index]}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=colors[index],
        )

    fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/reduced_cov/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    sigma_nl_par = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    tracers = {"LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]], "ELG_LOP": [[0.8, 1.1], [1.1, 1.6]], "QSO": [[0.8, 2.1]]}
    nmocks = {"LRG": [0, 25], "ELG_LOP": [0, 10], "QSO": [7, 25]}
    sigma_nl_perp = {
        "LRG": [
            [3.0, 2.5],
            [4.0, 2.0],
            [5.0, 4.0],
        ],
        "ELG_LOP": [[6.0, 6.0], [3.0, 2.5]],
        "QSO": [[3.0, 0.0]],
    }
    sigma_s = {"LRG": [[2.0, 2.0], [2.0, 1.0], [2.0, 2.0]], "ELG_LOP": [[4.0, 7.0], [4.0, 3.0]], "QSO": [[4.0, 0.0]]}

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
                name = f"DESI_SecondGen_sm10_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
                dataset_xi = CorrelationFunction_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_dist=50.0,
                    max_dist=150.0,
                    realisation=None,
                    reduce_cov_factor=len(range(nmocks[t][0], nmocks[t][1])),
                    datafile=name,
                )

                # Loop over pre- and post-recon measurements
                for sig in range(len(sigma_nl_par)):

                    for n, n_poly in enumerate([[-2, 0, 2]]):

                        model = CorrBeutler2017(
                            recon=dataset_xi.recon,
                            isotropic=dataset_xi.isotropic,
                            marg="full",
                            fix_params=["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"],
                            poly_poles=dataset_xi.fit_poles,
                            correction=Correction.NONE,
                            n_poly=n_poly,
                        )
                        model.set_default("sigma_nl_par", sigma_nl_par[sig])
                        model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r])
                        model.set_default("sigma_s", sigma_s[t][i][r])

                        # Load in a pre-existing BAO template
                        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                        name = dataset_xi.name + f" mock mean fixed_type {sig} n_poly=" + str(n)
                        fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[i - 1])
                        allnames.append(name)

    # Submit all the jobs to NERSC. We have quite a few (231), so we'll
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
        plotnames = [f"{t.lower()}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
        datanames = [f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
        print(datanames)

        # Loop over all the chains
        stats = [[] for _ in range(len(datanames) * 2 - 1)]
        output_pre = {k: [] for k in datanames}
        output_post = {k: [] for k in datanames}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the tracer bin, sigma bin and n_poly bin
            print(extra["name"].split(" ")[3].lower())
            data_bin = datanames.index(extra["name"].split(" ")[3].lower())
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            sigma_bin = int(extra["name"].split("fixed_type ")[1].split(" ")[0])
            stats_bin = recon_bin * len(datanames) + data_bin
            print(extra["name"], data_bin, recon_bin, sigma_bin, stats_bin)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                    ]
                ],
                weight,
                axis=0,
            )

            stats[stats_bin].append([sigma_nl_par[sigma_bin], mean[0] - 1.0, mean[1] - 1.0, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])])
            if recon_bin == 0:
                output_pre[datanames[data_bin]].append(
                    f"{sigma_nl_par[sigma_bin]:6.4f}, {mean[0]-1.0:6.4f}, {mean[1]-1.0:6.4f}, {np.sqrt(cov[0, 0]):6.4f}, {np.sqrt(cov[1, 1]):6.4f}"
                )
            else:
                output_post[datanames[data_bin]].append(
                    f"{sigma_nl_par[sigma_bin]:6.4f}, {mean[0]-1.0:6.4f}, {mean[1]-1.0:6.4f}, {np.sqrt(cov[0, 0]):6.4f}, {np.sqrt(cov[1, 1]):6.4f}"
                )

        # Plot histograms of the errors and r_off
        figname = "/".join(pfn.split("/")[:-1]) + f"/Prerecon_alphas.png"
        plot_alphas(np.array(stats[: len(datanames)]), figname, plotnames)
        figname = "/".join(pfn.split("/")[:-1]) + f"/Postrecon_alphas.png"
        plot_alphas(np.array(stats[len(datanames) :]), figname, plotnames)
