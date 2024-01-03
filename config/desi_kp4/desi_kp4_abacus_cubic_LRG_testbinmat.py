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
    for include_binmat in [True, False]:

        dataset_xi = CorrelationFunction_DESI_KP4(
            recon=recon,
            fit_poles=[0, 2],
            min_dist=52.0,
            max_dist=150.0,
            realisation=None,
            num_mocks=1000,
            reduce_cov_factor=1,
            datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
        )

        model = CorrBeutler2017(
            recon=dataset_xi.recon,
            isotropic=dataset_xi.isotropic,
            marg="full",
            fix_params=["om"],
            poly_poles=dataset_xi.fit_poles,
            correction=Correction.HARTLAP,
            include_binmat=include_binmat,
            n_poly=[0, 2],
        )
        model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
        model.set_default("beta", 0.4, min=0.1, max=0.7)
        model.set_default("sigma_nl_par", sigma[recon][0], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
        model.set_default("sigma_nl_perp", sigma[recon][1], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
        model.set_default("sigma_s", sigma[recon][2], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

        # Load in a pre-existing BAO template
        pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
        model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

        name = dataset_xi.name + " mock mean include_binmat=" + str(include_binmat)
        fitter.add_model_and_dataset(model, dataset_xi, name=name)
        allnames.append(name)

        for j in range(len(dataset_xi.mock_data)):
            dataset_xi.set_realisation(j)
            name = dataset_xi.name + f" realisation {j} include_binmat=" + str(include_binmat)
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

        # Loop over all the fitters
        stats = [[] for _ in range(2)]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            poly_bin = 0 if extra["name"].split("include_binmat=")[1].split(" ")[0] == "True" else 1
            realisation = str(extra["name"].split()[-1]) if "realisation" in extra["name"] else "mean"
            print(extra["name"], poly_bin, realisation)

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

            stats[poly_bin].append(
                [
                    mean[0] - 1.0,
                    mean[1] - 1.0,
                    mean[2] - 1.0,
                    mean[3] - 1.0,
                    np.sqrt(cov[0, 0]),
                    np.sqrt(cov[1, 1]),
                    np.sqrt(cov[2, 2]),
                    np.sqrt(cov[3, 3]),
                    cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                    cov[2, 3] / np.sqrt(cov[2, 2] * cov[3, 3]),
                    params_dict["alpha"],
                    params_dict["epsilon"],
                ]
            )

        stats = np.array(stats)
        print(np.shape(stats))

        print((stats[1, 1:, [0, 1, 4, 5]] - stats[0, 1:, [0, 1, 4, 5]]).T)

        fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=1, sharex=True, squeeze=False)
        plt.subplots_adjust(left=0.1, top=0.95, bottom=0.08, right=0.95, hspace=0.0, wspace=0.0)

        boxprops = {"lw": 1.3, "color": "b"}
        medianprops = {"lw": 1.5, "color": "r"}
        whiskerprops = {"lw": 1.3, "color": "k"}
        axes[0, 0].boxplot(
            100.0 * (stats[1, 1:, [0, 1, 4, 5]] - stats[0, 1:, [0, 1, 4, 5]]).T,
            positions=np.arange(4),
            widths=0.4,
            whis=(0, 100),
            showfliers=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=whiskerprops,
        )

        axes[0, 0].axhline(0.0, color="k", ls="--", zorder=0, lw=0.8)
        axes[0, 0].set_ylabel("$\mathrm{With\,-\,Without\,Binning\,Matrix}$")
        plt.setp(
            axes,
            xticks=np.arange(4),
            xticklabels=[
                "$\\Delta\\alpha_{\mathrm{iso}} (\\%)$",
                "$\\Delta\\alpha_{\mathrm{ap}} (\\%)$",
                "$\\Delta\\alpha_{\mathrm{iso}} / \\sigma_{\\alpha_{\mathrm{iso}}}$",
                "$\\Delta\\alpha_{\mathrm{ap}} / \\sigma_{\\alpha_{\mathrm{ap}}}$",
            ],
        )
        plt.xticks(rotation=30)
        fig.savefig("/".join(pfn.split("/")[:-1]) + "/DESI_FirstGen_tesbinmat.png", bbox_inches="tight", transparent=True, dpi=300)
