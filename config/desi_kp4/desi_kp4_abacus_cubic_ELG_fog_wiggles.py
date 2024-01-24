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
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    sigma = {None: [9.5, 5.0, 2.0], "sym": [5.0, 2.0, 2.0]}

    # colors = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Loop over the mocktypes
    allnames = []

    # Loop over pre- and post-recon power spectrum measurements
    recon = "sym"
    for fog_wiggles in [True]:

        # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
        # First load up mock mean and add it to the fitting list.
        dataset_pk = PowerSpectrum_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=1,
            datafile="desi_kp4_abacus_cubicbox_cv_pk_elg.pkl",
        )

        for n, (broadband_type, n_poly) in enumerate(zip(["poly", "spline"], [[-1, 0, 1, 2, 3], 30])):

            model = PowerBeutler2017(
                recon=dataset_pk.recon,
                isotropic=dataset_pk.isotropic,
                fix_params=["om"],
                marg="full",
                poly_poles=dataset_pk.fit_poles,
                correction=Correction.HARTLAP,
                fog_wiggles=fog_wiggles,
                broadband_type=broadband_type,
                n_poly=n_poly,
            )
            model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
            model.set_default("beta", 0.4, min=0.1, max=0.7)
            model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
            model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

            # Load in a pre-existing BAO template
            pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
            model.kvals, model.pksmooth, model.pkratio = pktemplate.T

            name = dataset_pk.name + f" mock mean n_poly=" + str(n)
            fitter.add_model_and_dataset(model, dataset_pk, name=name)
            allnames.append(name)

            for j in range(len(dataset_pk.mock_data)):
                dataset_pk.set_realisation(j)
                name = dataset_pk.name + f" realisation {j} n_poly=" + str(n)
                fitter.add_model_and_dataset(model, dataset_pk, name=name)

        dataset_xi = CorrelationFunction_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_dist=52.0,
            max_dist=150.0,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=1,
            datafile="desi_kp4_abacus_cubicbox_cv_xi_elg.pkl",
        )

        for n, (broadband_type, n_poly) in enumerate(zip(["poly", "spline"], [[-2, -1, 0], [0, 2]])):

            model = CorrBeutler2017(
                recon=dataset_xi.recon,
                isotropic=dataset_xi.isotropic,
                marg="full",
                fix_params=["om"],
                poly_poles=dataset_xi.fit_poles,
                correction=Correction.HARTLAP,
                fog_wiggles=fog_wiggles,
                broadband_type=broadband_type,
                n_poly=n_poly,
            )
            model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
            model.set_default("beta", 0.4, min=0.1, max=0.7)
            model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
            model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
            model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

            # Load in a pre-existing BAO template
            pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
            model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

            name = dataset_xi.name + " mock mean n_poly=" + str(n)
            fitter.add_model_and_dataset(model, dataset_xi, name=name)
            allnames.append(name)

            for j in range(len(dataset_xi.mock_data)):
                dataset_xi.set_realisation(j)
                name = dataset_xi.name + f" realisation {j} n_poly=" + str(n)
                fitter.add_model_and_dataset(model, dataset_xi, name=name)

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

        datanames = ["fog_wiggles"]
        broadband_names = ["poly", "spline"]
        d_names = ["xi", "pk"]

        for dataname in datanames:
            for broadband in broadband_names:
                for d in d_names:
                    plotname = f"{dataname}_{broadband}_{d}"
                    dir_name = "/".join(pfn.split("/")[:-1]) + "/" + plotname
                    try:
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name, exist_ok=True)
                    except Exception:
                        pass

        # Loop over all the fitters
        c = [[[ChainConsumer() for _ in range(len(datanames))] for _ in range(len(broadband_names))] for d in range(len(d_names))]
        stats = [[[[] for _ in range(len(datanames))] for _ in range(len(broadband_names))] for d in range(len(d_names))]

        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            data_bin = 0 if "Xi" in extra["name"] else 1
            poly_bin = int(extra["name"].split("n_poly=")[1].split(" ")[0])
            dilate_bin = 0 if model.fog_wiggles else 1
            realisation = str(extra["name"].split()[-1]) if "realisation" in extra["name"] else "mean"
            print(extra["name"], data_bin, poly_bin, dilate_bin, realisation)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            df["$\\alpha_{ap}$"] = (1.0 + df["$\\epsilon$"].to_numpy()) ** 3
            newweight = np.where(
                np.logical_and(
                    np.logical_and(df["$\\alpha_\\parallel$"] >= 0.8, df["$\\alpha_\\parallel$"] <= 1.2),
                    np.logical_and(df["$\\alpha_\\perp$"] >= 0.8, df["$\\alpha_\\perp$"] <= 1.2),
                ),
                weight,
                0.0,
            )

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior[newweight > 0].argmax()
            params = df[newweight > 0].iloc[max_post]
            params_dict = model.get_param_dict(chain[newweight > 0][max_post])
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
                newweight,
                axis=0,
            )

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            if realisation == "mean":
                extra.pop("color", None)
                c[data_bin][poly_bin][dilate_bin].add_chain(
                    df, weights=newweight, color="k", **extra, plot_contour=True, plot_point=False, show_as_1d_prior=False
                )
                figname = None
                mean_mean, cov_mean = mean, cov
            else:
                c[data_bin][poly_bin][dilate_bin].add_marker(params, **extra)
                dataname = extra["name"].split(" ")[3].lower()
                plotname = f"{datanames[dilate_bin]}_{broadband_names[poly_bin]}_{d_names[data_bin]}"
                figname = "/".join(pfn.split("/")[:-1]) + "/" + plotname + "/" + extra["name"].replace(" ", "_") + "_contour.png"
                if not os.path.isfile(figname):
                    extra.pop("color", None)
                    # cc = ChainConsumer()
                    # cc.add_chain(df, weights=newweight, **extra)
                    # cc.add_marker(df.iloc[max_post], **extra)
                    # cc.plotter.plot(filename=figname)
                    figname = "/".join(pfn.split("/")[:-1]) + "/" + plotname + "/" + extra["name"].replace(" ", "_") + "_bestfit.png"
                else:
                    figname = None

            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(params_dict, display=False, figname=figname, title=extra["name"])
            if realisation == "mean":
                print(25.0 * new_chi_squared, dof)

            stats[data_bin][poly_bin][dilate_bin].append(
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
                    params_dict["alpha"],
                    params_dict["epsilon"],
                ]
            )

        for data_bin in range(len(d_names)):
            for poly_bin in range(len(broadband_names)):
                for dilate_bin in range(len(datanames)):
                    plotname = f"{datanames[dilate_bin]}_{broadband_names[poly_bin]}_{d_names[data_bin]}"

                    mean = np.mean(stats[data_bin][poly_bin][dilate_bin][1:], axis=0)
                    cov = np.cov(stats[data_bin][poly_bin][dilate_bin][1:], rowvar=False)

                    c[data_bin][poly_bin][dilate_bin].add_covariance(
                        mean[:4],
                        cov[:4, :4],
                        parameters=["$\\alpha$", "$\\alpha_{ap}$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                        plot_contour=True,
                        plot_point=False,
                        show_as_1d_prior=False,
                    )

                    truth = {
                        "$\\alpha$": 1.0,
                        "$\\alpha_{ap}$": 1.0,
                        "$\\alpha_\\perp$": 1.0,
                        "$\\alpha_\\parallel$": 1.0,
                    }

                    c[data_bin][poly_bin][dilate_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha_\\parallel$",
                            "$\\alpha_\\perp$",
                        ],
                        legend=False,
                    )
                    c[data_bin][poly_bin][dilate_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour2.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha$",
                            "$\\alpha_{ap}$",
                        ],
                        legend=False,
                    )

                    np.save("/".join(pfn.split("/")[:-1]) + "/Summary_" + plotname + f".npy", stats[data_bin][poly_bin][dilate_bin])
