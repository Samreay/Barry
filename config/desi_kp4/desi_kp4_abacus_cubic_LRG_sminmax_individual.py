import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import Optimiser
from barry.config import setup
from barry.models import CorrBeutler2017
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
def plot_grids_bias(stats, kmins, kmaxs, figname, inds, edgevals):

    dkmin = kmins[1] - kmins[0]
    dkmax = kmaxs[1] - kmaxs[0]

    statsmean = np.mean(stats, axis=0).T
    # bestmean = np.argmin(np.sqrt(statsmean[2] ** 2 + statsmean[3] ** 2))
    bestmean = np.where((stats[0][:, 0] == 46.0) & (stats[0][:, 1] == 154.0))[0][0]
    print(bestmean, statsmean[:, bestmean])

    fig, axes = plt.subplots(figsize=(7.5, 2.5), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.10, top=0.97, bottom=0.18, right=0.8, hspace=0.0, wspace=0.10)

    axes[0, 0].imshow(
        100.0 * (statsmean[inds[0]]).reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 4.0, kmins[-1] + 4.0, kmaxs[0] - 4.0, kmaxs[-1] + 4.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.25,
        vmax=0.25,
    )
    cax = axes[0, 1].imshow(
        100.0 * (statsmean[inds[1]]).reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 4.0, kmins[-1] + 4.0, kmaxs[0] - 4.0, kmaxs[-1] + 4.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.25,
        vmax=0.25,
    )
    lines = contour_rect(100.0 * (statsmean[inds[0]]).reshape(len(kmins), len(kmaxs)).T, edgevals[0])
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(100.0 * (statsmean[inds[1]]).reshape(len(kmins), len(kmaxs)).T, edgevals[1])
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(statsmean[0, bestmean], statsmean[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    axes[0, 1].errorbar(statsmean[0, bestmean], statsmean[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    fig.supxlabel(r"$s_{\mathrm{min}}\,(h^{-1}\,\mathrm{Mpc})$", x=0.45)
    fig.supylabel(r"$s_{\mathrm{max}}\,(h^{-1}\,\mathrm{Mpc})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=r"$\Delta \alpha_{\mathrm{iso},\mathrm{ap}}\,(\%)$" if inds[0] == 2 else r"$\Delta \alpha_{||,\perp}\,(\%)$",
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$" if inds[0] == 2 else r"$\alpha_{||}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$" if inds[1] == 3 else r"$\alpha_{\perp}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(kmins[0] - 4.4, kmins[-1] + 4.4)
    axes[0, 0].set_ylim(kmaxs[0] - 4.2, kmaxs[-1] + 4.4)
    axes[0, 1].set_xlim(kmins[0] - 4.4, kmins[-1] + 4.2)
    axes[0, 1].set_ylim(kmaxs[0] - 4.4, kmaxs[-1] + 4.4)

    fig.savefig(figname, bbox_inches="tight", transparent=False, dpi=300)


def plot_grids_errs(stats, kmins, kmaxs, figname, inds, edgevals):

    dkmin = kmins[1] - kmins[0]
    dkmax = kmaxs[1] - kmaxs[0]

    statsmean = np.mean(stats, axis=0).T
    statsstd = np.std(stats, axis=0).T
    # bestmean = np.argmin(np.sqrt(statsmean[2] ** 2 + statsmean[3] ** 2))
    bestmean = np.where((stats[0][:, 0] == 46.0) & (stats[0][:, 1] == 154.0))[0][0]

    statsstd /= statsstd[:, bestmean][:, None]

    fig, axes = plt.subplots(figsize=(7.5, 2.5), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.10, top=0.97, bottom=0.18, right=0.8, hspace=0.0, wspace=0.10)

    axes[0, 0].imshow(
        statsstd[inds[0]].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 4.0, kmins[-1] + 4.0, kmaxs[0] - 4.0, kmaxs[-1] + 4.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.50,
        vmax=1.50,
    )
    cax = axes[0, 1].imshow(
        statsstd[inds[1]].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 4.0, kmins[-1] + 4.0, kmaxs[0] - 4.0, kmaxs[-1] + 4.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.50,
        vmax=1.50,
    )
    lines = contour_rect(100.0 * (statsmean[inds[0]]).reshape(len(kmins), len(kmaxs)).T, edgevals[0])
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(100.0 * (statsmean[inds[1]]).reshape(len(kmins), len(kmaxs)).T, edgevals[1])
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(statsmean[0, bestmean], statsmean[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    axes[0, 1].errorbar(statsmean[0, bestmean], statsmean[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    fig.supxlabel(r"$s_{\mathrm{min}}\,(h^{-1}\,\mathrm{Mpc})$", x=0.35)
    fig.supylabel(r"$s_{\mathrm{max}}\,(h^{-1}\,\mathrm{Mpc})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=r"$\mathrm{Relative}\,\,\sigma_{\alpha_{\mathrm{iso},\mathrm{ap}}}$"
        if inds[0] == 2
        else r"$\mathrm{Relative}\,\,\sigma_{\alpha_{||,\perp}}$",
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$" if inds[0] == 2 else r"$\alpha_{||}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$" if inds[1] == 3 else r"$\alpha_{\perp}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(kmins[0] - 2.2, kmins[-1] + 2.2)
    axes[0, 0].set_ylim(kmaxs[0] - 2.2, kmaxs[-1] + 2.2)
    axes[0, 1].set_xlim(kmins[0] - 2.2, kmins[-1] + 2.2)
    axes[0, 1].set_ylim(kmaxs[0] - 2.2, kmaxs[-1] + 2.2)

    fig.savefig(figname, bbox_inches="tight", transparent=False, dpi=300)


def contour_rect(data, edgeval):

    im = np.where(np.fabs(data) <= edgeval, 1, 0)

    pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding

    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

    lines = []

    for ii, jj in np.ndindex(im0.shape):
        if im0[ii, jj] == 1:
            lines += [([ii - 0.5, ii - 0.5], [jj - 0.5, jj + 0.5])]
        if im1[ii, jj] == 1:
            lines += [([ii - 0.5, ii + 0.5], [jj - 0.5, jj - 0.5])]

    return lines


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = Optimiser(temp_dir=dir_name)

    # The optimal sigma values we found when fitting the mocks with fixed alpha/epsilon
    smins = np.array([22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0, 78.0])
    smaxs = np.array([122.0, 130.0, 138.0, 146.0, 154.0, 162.0, 170.0, 178.0])

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon measurements
    for r, recon in enumerate(["sym"]):

        model = CorrBeutler2017(recon=recon, fix_params=["om"], marg="full", poly_poles=[0, 2], correction=Correction.NONE, n_poly=[0, 2])
        model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
        model.set_default("beta", 0.4, min=0.1, max=0.7)
        model.set_default("sigma_nl_par", 5.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        model.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=1.0, prior="gaussian")
        model.set_default("sigma_s", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

        # Load in a pre-existing BAO template
        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

        for smin in smins:
            for smax in smaxs:

                dataset = CorrelationFunction_DESI_KP4(
                    datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
                    recon=model.recon,
                    fit_poles=model.poly_poles,
                    min_dist=smin,
                    max_dist=smax,
                    realisation=None,
                    num_mocks=1000,
                    reduce_cov_factor=1,
                )

                name = dataset.name + f" mock mean smin =" + str(smin) + " smax =" + str(smax)
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

                for j in range(len(dataset.mock_data)):
                    dataset.set_realisation(j)
                    name = dataset.name + f" realisation {j} smin =" + str(smin) + " smax =" + str(smax)
                    fitter.add_model_and_dataset(model, dataset, name=name)
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

        # Loop over all the chains
        stats = [[] for _ in range(len(dataset.mock_data))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            if "Prerecon" in extra["name"]:
                continue

            if "mock mean" in extra["name"]:
                continue

            # recon_bin = 0 if "Prerecon" in extra["name"] else 1
            kminbin = np.searchsorted(smins, extra["name"].split("smin =")[1].split(" ")[0])
            kmaxbin = np.searchsorted(smaxs, extra["name"].split("smax =")[1].split(" ")[0])
            realisation = int(extra["name"].split("realisation ")[1].split(" ")[0])

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels()).to_numpy()[0]

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df[1], df[2])
            print(extra["name"], kminbin, kmaxbin, smins[kminbin], smaxs[kmaxbin])

            stats[realisation].append(
                [
                    smins[kminbin],
                    smaxs[kmaxbin],
                    df[1] - 1.0,
                    (1.0 + df[2]) ** 3 - 1.0,
                    alpha_par - 1.0,
                    alpha_perp - 1.0,
                ]
            )

        print(np.shape(np.array(stats)))

        # Plot grids of alpha bias and alpha error as a function of smin and smax
        plot_grids_bias(np.array(stats), smins, smaxs, "/".join(pfn.split("/")[:-1]) + "/sminmax_bias_postrecon.png", [2, 3], [0.1, 0.2])
        plot_grids_errs(np.array(stats), smins, smaxs, "/".join(pfn.split("/")[:-1]) + "/sminmax_errs_postrecon.png", [2, 3], [0.1, 0.2])
        plot_grids_bias(np.array(stats), smins, smaxs, "/".join(pfn.split("/")[:-1]) + "/sminmax_bias_postrecon2.png", [4, 5], [0.1, 0.1])
        plot_grids_errs(np.array(stats), smins, smaxs, "/".join(pfn.split("/")[:-1]) + "/sminmax_errs_postrecon2.png", [4, 5], [0.1, 0.1])
