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
import matplotlib.colors as mplc
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer


def plot_alphas(data, figname, plotnames):

    colors = [mplc.cnames[color] for color in ["orange", "orangered", "firebrick", "lightskyblue", "steelblue", "seagreen", "black"]]

    # Split up Pk and Xi. Further split these up across the different tracers
    fig = plt.figure(figsize=(12, 6))
    axes = gridspec.GridSpec(2, 1, figure=fig, left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.2, wspace=0.0)
    axes1 = axes[0, 0].subgridspec(2, np.shape(data)[0], hspace=0.0, wspace=0.0)  # Xi
    axes2 = axes[1, 0].subgridspec(2, np.shape(data)[0], hspace=0.0, wspace=0.0)  # Pk

    # Further split up Xi into different polynomial types
    for index in range(np.shape(data)[0]):
        print(plotnames[index])
        ax1_iso = fig.add_subplot(axes1[0, index])
        ax1_ap = fig.add_subplot(axes1[1, index])
        ax2_iso = fig.add_subplot(axes2[0, index])
        ax2_ap = fig.add_subplot(axes2[1, index])

        ax1_iso.plot(data[index, 0, :, 0], data[index, 0, :, 1], color=colors[index], zorder=1, alpha=0.75, lw=0.8)
        ax1_ap.plot(data[index, 0, :, 0], data[index, 0, :, 2], color=colors[index], zorder=1, alpha=0.75, lw=0.8)
        ax2_iso.plot(data[index, 1, :, 0], data[index, 1, :, 1], color=colors[index], zorder=1, alpha=0.75, lw=0.8)
        ax2_ap.plot(data[index, 1, :, 0], data[index, 1, :, 2], color=colors[index], zorder=1, alpha=0.75, lw=0.8)

        ax1_iso.fill_between(
            data[index, 0, :, 0],
            (data[index, 0, :, 1] - data[index, 0, :, 3]),
            (data[index, 0, :, 1] + data[index, 0, :, 3]),
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax1_ap.fill_between(
            data[index, 0, :, 0],
            (data[index, 0, :, 2] - data[index, 0, :, 4]),
            (data[index, 0, :, 2] + data[index, 0, :, 4]),
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax2_iso.fill_between(
            data[index, 1, :, 0],
            (data[index, 1, :, 1] - data[index, 1, :, 3]),
            (data[index, 1, :, 1] + data[index, 1, :, 3]),
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax2_ap.fill_between(
            data[index, 1, :, 0],
            (data[index, 1, :, 2] - data[index, 1, :, 4]),
            (data[index, 1, :, 2] + data[index, 1, :, 4]),
            color=colors[index],
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax1_iso.set_ylim(-1.0, 1.0)
        ax1_ap.set_ylim(-4.0, 4.0)
        ax2_iso.set_ylim(-1.0, 1.0)
        ax2_ap.set_ylim(-4.0, 4.0)
        ax2_ap.set_xlabel(r"$\Sigma_{s}$")
        if index == 0:
            ax1_iso.set_ylabel(r"$\alpha_{\mathrm{iso}} - 1\,(\%)$")
            ax1_ap.set_ylabel(r"$\alpha_{\mathrm{ap}} - 1\,(\%)$")
            ax2_iso.set_ylabel(r"$\alpha_{\mathrm{iso}} - 1\,(\%)$")
            ax2_ap.set_ylabel(r"$\alpha_{\mathrm{ap}} - 1\,(\%)$")
        else:
            ax1_iso.set_yticklabels([])
            ax1_ap.set_yticklabels([])
            ax2_iso.set_yticklabels([])
            ax2_ap.set_yticklabels([])
        ax1_iso.set_xticklabels([])
        ax1_ap.set_xticklabels([])
        ax2_iso.set_xticklabels([])
        for val, ls in zip([-0.3, 0.0, 0.3], [":", "--", ":"]):
            ax1_iso.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
            ax1_ap.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
            ax2_iso.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
            ax2_ap.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
        ax1_iso.axvline(2.0, color="k", ls=":", zorder=0, lw=0.8)
        ax1_ap.axvline(2.0, color="k", ls=":", zorder=0, lw=0.8)
        ax2_iso.axvline(2.0, color="k", ls=":", zorder=0, lw=0.8)
        ax2_ap.axvline(2.0, color="k", ls=":", zorder=0, lw=0.8)
        ax1_iso.text(
            0.05,
            0.95,
            f"{plotnames[index]}",
            transform=ax1_iso.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=colors[index],
        )
        if index == 6:
            ax1_iso.text(
                0.95,
                0.15,
                f"$\\xi(s)$",
                transform=ax1_iso.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="k",
            )
            ax2_iso.text(
                0.95,
                0.15,
                f"$P(k)$",
                transform=ax2_iso.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="k",
            )

    fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)


if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/reduced_cov/")

    # Set up the Fitting class and sampler
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    colors = [mplc.cnames[color] for color in ["orange", "orangered", "firebrick", "lightskyblue", "steelblue", "seagreen", "black"]]

    tracers = {
        "LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
        "ELG_LOP": [[0.8, 1.1], [1.1, 1.6]],
        "QSO": [[0.8, 2.1]],
        "BGS_BRIGHT-21.5": [[0.1, 0.4]],
    }
    reconsmooth = {"LRG": 10, "ELG_LOP": 10, "QSO": 30, "BGS_BRIGHT-21.5": 15}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [9.0, 6.0],
        ],
        "ELG_LOP": [[8.5, 6.0], [8.5, 6.0]],
        "QSO": [[9.0, 6.0]],
        "BGS_BRIGHT-21.5": [[10.0, 8.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [4.5, 3.0],
            [4.5, 3.0],
            [4.5, 3.0],
        ],
        "ELG_LOP": [[4.5, 3.0], [4.5, 3.0]],
        "QSO": [[3.5, 3.0]],
        "BGS_BRIGHT-21.5": [[6.5, 3.0]],
    }
    sigma_s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = (
        "default_FKP"
        # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    )
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"

    count = 0
    plotnames = [f"{t}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
    datanames = [f"{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for r, recon in enumerate(["sym"]):

                # Correlation function data
                name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_xi.pkl"
                dataset_xi = CorrelationFunction_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_dist=50.0,
                    max_dist=150.0,
                    realisation=None,
                    reduce_cov_factor=25,
                    datafile=name,
                )

                # Power spectrum data
                name = f"DESI_SecondGen_sm{reconsmooth[t]}_{t.lower()}_{ffa}_{cap}_{zs[0]}_{zs[1]}_{rp}_pk.pkl"
                dataset_pk = PowerSpectrum_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_k=0.02,
                    max_k=0.30,
                    realisation=None,
                    reduce_cov_factor=25,
                    datafile=name,
                )

                for sig in range(len(sigma_s)):

                    # Correlation Function model
                    model = CorrBeutler2017(
                        recon=recon,
                        isotropic=False,
                        marg="full",
                        fix_params=["om"],
                        poly_poles=dataset_xi.fit_poles,
                        correction=Correction.NONE,
                        broadband_type="spline",
                        n_poly=[0, 2],
                    )
                    model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                    model.set_default("beta", 0.4, min=0.1, max=0.7)
                    model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                    model.set_default("sigma_s", sigma_s[sig], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                    # Load in a pre-existing BAO template
                    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                    name = dataset_xi.name + f" mock mean fixed_type {sig}"
                    fitter.add_model_and_dataset(model, dataset_xi, name=name, color=colors[count])

                    # Power spectrum model
                    model = PowerBeutler2017(
                        recon=recon,
                        isotropic=False,
                        marg="full",
                        fix_params=["om"],
                        poly_poles=dataset_pk.fit_poles,
                        correction=Correction.NONE,
                        broadband_type="spline",
                        n_poly=30,
                    )
                    model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                    model.set_default("beta", 0.4, min=0.1, max=0.7)
                    model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                    model.set_default("sigma_s", sigma_s[sig], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                    # Load in a pre-existing BAO template
                    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                    model.kvals, model.pksmooth, model.pkratio = pktemplate.T

                    name = dataset_pk.name + f" mock mean fixed_type {sig}"
                    fitter.add_model_and_dataset(model, dataset_pk, name=name, color=colors[count])

            count += 1

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
        stats = [[[] for _ in range(2)] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            # Get the tracer bin, sigma bin and n_poly bin
            data_bin = datanames.index(extra["name"].split(" ")[3].lower())
            xi_bin = 0 if "Corr" in model.name else 1
            sigma_bin = int(extra["name"].split("fixed_type ")[1].split(" ")[0])
            print(extra["name"], data_bin, xi_bin, sigma_bin)

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
            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha$",
                        "$\\alpha_{ap}$",
                    ]
                ],
                newweight,
                axis=0,
            )

            stats[data_bin][xi_bin].append(
                [
                    sigma_s[sigma_bin],
                    100.0 * (mean[0] - 1.0),
                    100.0 * (mean[1] - 1.0),
                    100.0 * np.sqrt(cov[0, 0]),
                    100.0 * np.sqrt(cov[1, 1]),
                ]
            )

        stats = np.array(stats)

        # Plot histograms of the errors and r_off
        figname = "/".join(pfn.split("/")[:-1]) + f"/Postrecon_SecondGen_alphas_vs_sigma_s.png"
        plot_alphas(stats, figname, plotnames)
