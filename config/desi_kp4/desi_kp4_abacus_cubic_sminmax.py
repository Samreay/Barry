import sys

sys.path.append("..")
sys.path.append("../../")
from barry.samplers import DynestySampler
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
def plot_grids(stats, figname):

    fig, axes = plt.subplots(figsize=(5, 5), nrows=2, ncols=2, sharex=True, sharey="row", squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)

    axes[0, 0].imshow(stats[2])
    axes[0, 1].imshow(stats[7])
    axes[1, 0].imshow(stats[3])
    axes[1, 1].imshow(stats[8])
    plt.show()

    # fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)


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

    smins = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70])
    smaxs = np.array([130, 135, 140, 145, 150, 155, 160, 165, 170])

    colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon measurements
    for r, recon in enumerate([None, "sym"]):

        model = CorrBeutler2017(
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
        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

        for smin in smins:
            for smax in smaxs:

                dataset = CorrelationFunction_DESI_KP4(
                    recon=model.recon,
                    fit_poles=model.poly_poles,
                    min_dist=smin,
                    max_dist=smax,
                    realisation=None,
                    num_mocks=1000,
                    reduce_cov_factor=25,
                )

                name = dataset.name + f" mock mean smin =" + str(smin) + " smax =" + str(smax)
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

        # Set up a ChainConsumer instance. Plot the MAP for individual realisations and a contour for the mock average

        # Loop over all the chains
        stats = []
        output = []
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            print(extra["name"])
            if "Prerecon" in extra["name"]:
                continue
            # recon_bin = 0 if "Prerecon" in extra["name"] else 1
            sminbin = np.where(int(extra["name"].split("fixed_type ")[1].split(" ")[0]) == smins)[0][0]
            smaxbin = np.where(int(extra["name"].split("fixed_type ")[1].split(" ")[0]) == smins)[0][0]

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
            print(extra["name"], mean, cov)

            stats.append(
                [
                    smins[sminbin],
                    smaxs[smaxbin],
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
                f"{smins[sminbin]:6.4f}, {smaxs[smaxbin]:6.4f}, {mean[0]:6.4f}, {mean[1]:6.4f}, {mean[2]:6.4f}, {mean[3]:6.4f}, {mean[4]:6.4f}, {np.sqrt(cov[0, 0]):6.4f}, {np.sqrt(cov[1, 1]):6.4f}, {np.sqrt(cov[2, 2]):6.4f}, {np.sqrt(cov[3, 3]):6.4f}, {np.sqrt(cov[4, 4]):6.4f}"
            )

        print(stats)

        # Plot grids of alpha bias and alpha error as a function of smin and smax
        plot_grids(stats, "/".join(pfn.split("/")[:-1]) + "/sminmax_postrecon_npoly4.png")

        # Save all the numbers to a file
        with open(dir_name + "/Barry_fit_sminmax_postrecon_npoly4.txt", "w") as f:
            f.write(
                "# smin,  smax,  alpha_par, alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof\n"
            )
            for l in output:
                f.write(l + "\n")
