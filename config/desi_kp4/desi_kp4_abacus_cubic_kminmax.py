import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.

# Convenience function to plot histograms of the errors and cross-correlation coefficients
def plot_grids_bias(stats, kmins, kmaxs, figname):

    fig, axes = plt.subplots(figsize=(5, 3), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.15, top=0.97, bottom=0.17, right=0.8, hspace=0.0, wspace=0.10)

    print((stats[2] / stats[7]).reshape(len(kmins), len(kmaxs)).T)
    print((stats[3] / stats[8]).reshape(len(kmins), len(kmaxs)).T)
    axes[0, 0].imshow(
        (stats[2] / stats[7]).reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] + 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-2.5,
        vmax=2.5,
    )
    cax = axes[0, 1].imshow(
        (stats[3] / stats[8]).reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] + 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-2.5,
        vmax=2.5,
    )
    fig.supxlabel(r"$k_{\mathrm{min}}\,(h\,\mathrm{Mpc}^{-1})$", x=0.45)
    fig.supylabel(r"$k_{\mathrm{max}}\,(h\,\mathrm{Mpc}^{-1})$", y=0.55)
    fig.colorbar(cax, ax=axes.ravel().tolist(), label=r"$\Delta \alpha_{||,\perp} / \sigma_{\alpha_{||,\perp}} $")
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{||}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\perp}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )

    fig.savefig(figname, bbox_inches="tight", transparent=False, dpi=300)


def plot_grids_errs(stats, kmins, kmaxs, figname):

    fig, axes = plt.subplots(figsize=(5, 3), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.15, top=0.97, bottom=0.17, right=0.8, hspace=0.0, wspace=0.10)

    print(stats[7].reshape(len(kmins), len(kmaxs)).T)
    print(stats[8].reshape(len(kmins), len(kmaxs)).T)
    axes[0, 0].imshow(
        100.0 * stats[7].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] + 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        vmin=0.05,
        vmax=0.2,
    )
    cax = axes[0, 1].imshow(
        100.0 * stats[8].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] + 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        vmin=0.05,
        vmax=0.2,
    )
    fig.supxlabel(r"$k_{\mathrm{min}}\,(h\,\mathrm{Mpc}^{-1})$", x=0.45)
    fig.supylabel(r"$k_{\mathrm{max}}\,(h\,\mathrm{Mpc}^{-1})$", y=0.55)
    fig.colorbar(cax, ax=axes.ravel().tolist(), label=r"$\sigma_{\alpha_{||,\perp}}\,(\%)$")
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{||}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\perp}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )

    fig.savefig(figname, bbox_inches="tight", transparent=False, dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/reduced_cov/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=250)

    # The optimal sigma values we found when fitting the mocks with fixed alpha/epsilon
    sigma_nl_par = {None: 9.6, "sym": 5.4}
    sigma_nl_perp = {None: 5.0, "sym": 1.8}
    sigma_s = 0.0

    kmins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    kmaxs = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon measurements
    for r, recon in enumerate([None, "sym"]):

        model = PowerBeutler2017(
            recon=recon,
            marg="full",
            poly_poles=[0, 2],
            correction=Correction.NONE,
            n_poly=4,  # 4 polynomial terms for Xi(s)
        )

        # Set Gaussian priors for the BAO damping centred on the optimal values
        # found from fitting with fixed alpha/epsilon and with width 2 Mpc/h
        model.set_default("sigma_nl_par", sigma_nl_par[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        model.set_default("sigma_nl_perp", sigma_nl_perp[recon], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        model.set_default("sigma_s", sigma_s, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
        model.kvals, model.pksmooth, model.pkratio = pktemplate.T

        for kmin in kmins:
            for kmax in kmaxs:

                dataset = PowerSpectrum_DESI_KP4(
                    datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
                    recon=model.recon,
                    fit_poles=model.poly_poles,
                    min_k=kmin,
                    max_k=kmax,
                    realisation=None,
                    num_mocks=1000,
                    reduce_cov_factor=25,
                )

                name = dataset.name + f" mock mean kmin =" + str(kmin) + " kmax =" + str(kmax)
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
        stats = []
        output = []
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            print(extra["name"])
            if "Prerecon" in extra["name"]:
                continue
            # recon_bin = 0 if "Prerecon" in extra["name"] else 1
            kminbin = np.searchsorted(kmins, extra["name"].split("kmin =")[1].split(" ")[0])
            kmaxbin = np.searchsorted(kmaxs, extra["name"].split("kmax =")[1].split(" ")[0])

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

            print(extra["name"], kminbin, kmaxbin, kmins[kminbin], kmaxs[kmaxbin])

            stats.append(
                [
                    kmins[kminbin],
                    kmaxs[kmaxbin],
                    mean[0] - 1.0,
                    mean[1] - 1.0,
                    mean[2] - 5.4,
                    mean[3] - 1.8,
                    mean[4],
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    np.sqrt(cov[3, 3]),
                    np.sqrt(cov[4, 4]),
                ]
            )
            output.append(
                f"{kmins[kminbin]:6.4f}, {kmaxs[kmaxbin]:6.4f}, {mean[0]:6.4f}, {mean[1]:6.4f}, {mean[2]:6.4f}, {mean[3]:6.4f}, {mean[4]:6.4f}, {np.sqrt(cov[0, 0]):6.4f}, {np.sqrt(cov[1, 1]):6.4f}, {np.sqrt(cov[2, 2]):6.4f}, {np.sqrt(cov[3, 3]):6.4f}, {np.sqrt(cov[4, 4]):6.4f}"
            )

        print(stats)

        # Plot grids of alpha bias and alpha error as a function of smin and smax
        plot_grids_bias(np.array(stats).T, kmins, kmaxs, "/".join(pfn.split("/")[:-1]) + "/kminmax_bias_postrecon_npoly6.png")
        plot_grids_errs(np.array(stats).T, kmins, kmaxs, "/".join(pfn.split("/")[:-1]) + "/kminmax_errs_postrecon_npoly6.png")

        # Save all the numbers to a file
        with open(dir_name + "/Barry_fit_sminmax_postrecon_npoly4.txt", "w") as f:
            f.write(
                "# smin,  smax,  alpha_par, alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof\n"
            )
            for l in output:
                f.write(l + "\n")
