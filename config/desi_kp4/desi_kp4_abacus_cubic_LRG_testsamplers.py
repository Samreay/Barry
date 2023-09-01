import sys
import copy

sys.path.append("..")
sys.path.append("../../")
import time
from barry.samplers import NautilusSampler, ZeusSampler, EnsembleSampler, DynestySampler
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
from chainconsumer import ChainConsumer

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting classes
    fitter = Fitter(dir_name, remove_output=False)
    samplers = [
        NautilusSampler(temp_dir=dir_name),
        ZeusSampler(temp_dir=dir_name),
        EnsembleSampler(temp_dir=dir_name),
        DynestySampler(temp_dir=dir_name),
        DynestySampler(temp_dir=dir_name, dynamic=True),
    ]
    sampler_names = [
        r"$\mathrm{Nautilus}$",
        r"$\mathrm{Zeus}$",
        r"$\mathrm{Emcee}$",
        r"$\mathrm{Dynesty\,Static}$",
        r"$\mathrm{Dynesty\,Dynamic}$",
    ]

    # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
    # First load up mock mean and add it to the fitting list.
    dataset_pk = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1,
        datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
    )

    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1,
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
    )

    # We'll do this test using post-recon measurements only
    model_pk = PowerBeutler2017(
        recon=dataset_pk.recon,
        isotropic=dataset_pk.isotropic,
        fix_params=["om"],
        marg="full",
        poly_poles=dataset_pk.fit_poles,
        correction=Correction.NONE,
        n_poly=6,
    )
    model_pk.set_default("sigma_nl_par", 5.1, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_pk.set_default("sigma_nl_perp", 1.6, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_pk.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_pk.kvals, model_pk.pksmooth, model_pk.pkratio = pktemplate.T

    model_xi = CorrBeutler2017(
        recon=dataset_xi.recon,
        isotropic=dataset_xi.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset_xi.fit_poles,
        correction=Correction.NONE,
        n_poly=4,
    )
    model_xi.set_default("sigma_nl_par", 5.1, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_xi.set_default("sigma_nl_perp", 1.6, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_xi.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_xi.parent.kvals, model_xi.parent.pksmooth, model_xi.parent.pkratio = pktemplate.T

    for m, d in zip([model_pk, model_xi], [dataset_pk, dataset_xi]):
        d.set_realisation(None)
        name = d.name + f" mock mean"
        fitter.add_model_and_dataset(m, d, name=name)
        for j in range(len(d.mock_data)):
            d.set_realisation(j)
            name = d.name + f" realisation {j}"
            fitter.add_model_and_dataset(m, d, name=name)

    fitter.set_num_walkers(1)

    # Submit all the jobs
    sampler_index = 0
    fitter.set_sampler(samplers[sampler_index])
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
                data_bin = 0 if "Xi" in extra["name"] else 1

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
