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
import matplotlib as plt
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 100 live points.
    fitter = Fitter(dir_name, remove_output=True)
    sampler = DynestySampler(temp_dir=dir_name, nlive=250)

    mocktypes = ["abacus_cutsky"]
    nzbins = [3]

    # Loop over the mocktypes
    allnames = []
    for i, (mocktype, redshift_bins) in enumerate(zip(mocktypes, nzbins)):

        # Loop over the available redshift bins for each mock type
        for z in range(redshift_bins):

            # Create the data. We'll fit mono-, quad- and hexadecapole between k=0.02 and 0.3.
            # First load up mock mean and add it to the fitting list.
            dataset = CorrelationFunction_DESI_KP4(
                recon=None,
                fit_poles=[0, 2, 4],
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
                marg="full",
                fix_params=["om", "beta"],
                poly_poles=dataset.fit_poles,
                correction=Correction.HARTLAP,
            )

            # Create a unique name for the fit and add it to the list
            name = dataset.name + " mock mean"
            fitter.add_model_and_dataset(model, dataset, name=name)
            allnames.append(name)

            # Now add the individual realisations to the list
            for i in range(len(dataset.mock_data)):
                dataset.set_realisation(i)
                name = dataset.name + f" realisation {i}"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

    # Submit all the jobs to NERSC. We have quite a few (78), so we'll
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
            print(params_dict)

            # Get some useful properties of the fit, and plot the MAP if it's the mock mean
            figname = pfn + "_" + extra["name"].replace(" ", "_") + "_bestfit.pdf" if realisation == "mean" else None
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
            print(fitname, redshift_bin, [np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), corr])
            stats[fitname[redshift_bin]].append([np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), corr])
            output[fitname[redshift_bin]].append(
                f"{realisation:s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {new_chi_squared:7.3f}, {dof:4d}"
            )

        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        for z in range(nzbins[0]):
            c[z].configure(bins=20)
            c[z].plotter.plot(
                filename=[pfn + "_" + fitname[z] + "_contour.pdf"],
                truth=truth,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                legend=False,
            )
            print(c[z].analysis.get_correlations())

            # Save all the numbers to a file
            with open(dir_name + "/Barry_fit_" + fitname[z] + ".txt", "w") as f:
                f.write(
                    "# Realisation, alpha_par, alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof\n"
                )
                for l in output[fitname[z]]:
                    f.write(l + "\n")
