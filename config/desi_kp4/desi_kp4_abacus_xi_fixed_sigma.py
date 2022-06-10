import sys

sys.path.append("..")
sys.path.append("../..")
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
def plot_errors(stats, figname):

    nstats = len(stats)
    means = np.mean(stats, axis=0)
    covs = np.cov(stats, rowvar=False)
    corr = covs[0, 1] / np.sqrt(covs[0, 0] * covs[1, 1])

    labels = [r"$\sigma_{\alpha,||}$", r"$\sigma_{\alpha,\perp}$", r"$\rho(\alpha_{||},\alpha_{\perp})$"]
    colors = ["r", "b", "g"]
    fig, axes = plt.subplots(figsize=(7, 2), nrows=1, ncols=3, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.3)
    for ax, vals, avgs, stds, l, c in zip(
        axes.T, np.array(stats).T[2:5], means[2:5], [np.sqrt(covs[0, 0]), np.sqrt(covs[1, 1]), corr], labels, colors
    ):

        ax[0].hist(vals, 10, color=c, histtype="stepfilled", alpha=0.2, density=False, zorder=0)
        ax[0].hist(vals, 10, color=c, histtype="step", alpha=1.0, lw=1.3, density=False, zorder=1)
        ax[0].axvline(avgs, color="k", ls="--", zorder=2)
        ax[0].axvline(stds, color="k", ls=":", zorder=2)
        ax[0].set_xlabel(l)

    axes[0, 0].set_ylabel(r"$N_{\mathrm{mocks}}$")

    fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)

    return nstats, means, covs, corr


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=250)

    mocktypes = ["abacus_cutsky"]
    nzbins = [3]
    sigma_nl_perp = [6.0, 5.0, 4.0]
    sigma_nl_par = [10.0, 9.0, 8.0]

    # Loop over the mocktypes
    allnames = []
    for i, (mocktype, redshift_bins) in enumerate(zip(mocktypes, nzbins)):

        # Loop over the available redshift bins for each mock type
        for z in range(redshift_bins):

            # Loop over pre- and post-recon measurements
            for recon in [None]:

                # Create the data. We'll fit mono-, quad- and hexadecapole between k=0.02 and 0.3.
                # First load up mock mean and add it to the fitting list.
                dataset = CorrelationFunction_DESI_KP4(
                    recon=None,
                    fit_poles=[0, 2],
                    min_dist=50.0,
                    max_dist=170.0,
                    mocktype=mocktype,
                    redshift_bin=z + 1,
                    realisation=None,
                    num_mocks=1000,
                )

                # Set up the model we'll use. Fix Omega_m and beta. 5 polynomials (default)
                # for each of the fitted multipoles. Use full analytic marginalisation for speed
                # Apply the Hartlap correction to the covariance matrix.
                model = CorrBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    fix_params=["om", "beta", "sigma_nl_perp", "sigma_nl_perp", "sigma_fog"],
                    marg="partial",
                    poly_poles=dataset.fit_poles,
                    correction=Correction.HARTLAP,
                )
                model.set_default("sigma_nl_par", sigma_nl_par[i])
                model.set_default("sigma_nl_perp", sigma_nl_perp[i])
                model.set_default("sigma_s", 3.0)

                # Create a unique name for the fit and add it to the list
                name = dataset.name + " mock mean"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

                # Now add the individual realisations to the list
                for j in range(len(dataset.mock_data)):
                    dataset.set_realisation(j)
                    name = dataset.name + f" realisation {j}"
                    fitter.add_model_and_dataset(model, dataset, name=name)
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

        # Set up some ChainConsumer instances. We'll do one plot per redshift bin,
        # and plot the MAP for individual realisations and a contour for the mock average
        fitname = []
        zmins = ["0.4", "0.6", "0.8"]

        c = [ChainConsumer(), ChainConsumer(), ChainConsumer()]

        # Loop over all the chains
        stats = {}
        output = {}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the realisation number and redshift bin
            # recon_bin = 0 if "Prerecon" in extra["name"] else 1
            # redshift_bin = [recon_bin + 2 * i for i, zmin in enumerate(zmins) if zmin in extra["name"].split("_")[1]][0]
            redshift_bin = [i for i, zmin in enumerate(zmins) if zmin in extra["name"].split("_")[1]][0]
            realisation = str(extra["name"].split()[-1]) if "realisation" in extra["name"] else "mean"

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior.argmax()
            params = df.loc[max_post]
            params_dict = model.get_param_dict(chain[max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)

            # Get some useful properties of the fit, and plot the MAP if it's the mock mean
            figname = (
                "/".join(pfn.split("/")[:-1]) + "/" + extra["name"].replace(" ", "_") + "_bestfit.png"
                if realisation == "mean" or realisation == "10" or realisation == "23"
                else None
            )
            new_chi_squared, dof, bband, mods, smooths = model.plot(params_dict, display=False, figname=figname)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if realisation == "mean":
                fitname.append(data[0]["name"].replace(" ", "_"))
                stats[fitname[redshift_bin]] = []
                output[fitname[redshift_bin]] = []
                c[redshift_bin].add_chain(df, weights=weight, **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False)
            else:
                c[redshift_bin].add_marker(params, **extra)

            # Compute some summary statistics and add them to a dictionary
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

            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            stats[fitname[redshift_bin]].append([mean[0], mean[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), corr, new_chi_squared])
            output[fitname[redshift_bin]].append(
                f"{realisation:s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {new_chi_squared:7.3f}, {dof:4d}"
            )

        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        for z in range(3):
            c[z].configure(bins=20)
            c[z].plotter.plot(
                filename=["/".join(pfn.split("/")[:-1]) + "/" + fitname[z] + "_contour.png"],
                truth=truth,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                legend=False,
            )

            # Plot histograms of the errors and r_off
            nstats, means, covs, corr = plot_errors(stats[fitname[z]], "/".join(pfn.split("/")[:-1]) + "/" + fitname[z] + "_errors.png")

            # Save all the numbers to a file
            with open(dir_name + "/Barry_fit_" + fitname[z] + ".txt", "w") as f:
                f.write(
                    "# Realisation, alpha_par, alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof\n"
                )
                for l in output[fitname[z]]:
                    f.write(l + "\n")

                # And now the average of all the individual realisations
                f.write("# ---------------------------------------------------\n")
                f.write(
                    "# <alpha_par>, <alpha_perp>, <sigma_alpha_par>, <sigma_alpha_perp>, <corr_alpha_par_perp>, std_alpha_par, std_alpha_perp, corr_alpha_par_perp, <bf_chi2>\n"
                )
                f.write(
                    f"{means[0]:6.4f}, {means[1]:6.4f}, {means[2]:6.4f}, {means[3]:6.4f}, {means[4]:6.4f}, {np.sqrt(covs[0,0]):6.4f}, {np.sqrt(covs[1,1]):6.4f}, {corr:6.4f}, {means[5]:7.3f}\n"
                )
