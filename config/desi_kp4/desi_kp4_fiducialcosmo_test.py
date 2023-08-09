import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import Optimiser
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.fitter import Fitter

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Optimiser sampler.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = Optimiser(temp_dir=dir_name)

    # Specify the datafiles to fit
    datafiles = [
        "cubicbox_pk_lrg_abacussummit_c000_grid_c000.pkl",
        "cubicbox_pk_lrg_abacussummit_c000_grid_c001.pkl",
        "cubicbox_pk_lrg_abacussummit_c000_grid_c002.pkl",
        "cubicbox_pk_lrg_abacussummit_c000_grid_c003.pkl",
    ]

    # Set up the models pre and post recon
    model_pre = PowerBeutler2017(
        recon=None,
        isotropic=False,
        marg="full",
        poly_poles=[0, 2],
        n_poly=5,
    )
    model_post = PowerBeutler2017(
        recon="sym",
        isotropic=False,
        marg="full",
        poly_poles=[0, 2],
        n_poly=5,
    )

    # Loop over the datafiles and fit each mock realisation in the pairs
    allnames = []
    for i, datafile in enumerate(datafiles):

        # Loop over pre- and post-recon measurements
        for recon in [None, "sym"]:

            # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
            # First load up mock mean and add it to the fitting list. Use only the diagonal parts
            # of the covariance matrix
            dataset = PowerSpectrum_DESI_KP4(
                recon=recon,
                fit_poles=[0, 2],
                min_k=0.02,
                max_k=0.30,
                realisation=None,
                num_mocks=1000,
                datafile=datafile,
            )

            model = model_pre if recon is None else model_post

            # Now add the individual realisations to the list
            for j in range(len(dataset.mock_data)):
                dataset.set_realisation(j)
                name = dataset.name + f" realisation {j}"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

    # Submit all the job. We have quite a few (52), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    results = {}
    for datafile in datafiles:
        dataname = datafile.split("_")[6].split(".")[0]
        results[dataname] = {}
        for recon in ["Prerecon", "RecSym"]:
            results[dataname][recon] = np.empty((25, 4))

    alpha_par_trues = {"c000": 1.0, "c001": 1.01560850, "c002": 1.01431340, "c003": 1.00447187}
    alpha_perp_trues = {"c000": 1.0, "c001": 1.03502325, "c002": 0.99011363, "c003": 1.01305502}

    if fitter.should_plot():
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            dataname = extra["name"].split(" ")[6]
            recon_bin = "Prerecon" if "Prerecon" in extra["name"] else "RecSym"
            realisation = int(extra["name"].split(" ")[-1])

            df = pd.DataFrame(chain, columns=model.get_labels())

            alpha_par_true = alpha_par_trues[dataname]
            alpha_perp_true = alpha_perp_trues[dataname]
            alpha_true, epsilon_true = model.get_reverse_alphas(alpha_par_true, alpha_perp_true)

            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par - alpha_par_true
            df["$\\alpha_\\perp$"] = alpha_perp - alpha_perp_true
            df["$\\alpha$"] -= alpha_true  # For easier plotting
            df["$\\epsilon$"] -= epsilon_true  # For easier plotting

            results[dataname][recon_bin][realisation] = df[
                ["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"]
            ].to_numpy()

    # Now we have all the results, we can plot them
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex="row", squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.98, bottom=0.10, right=0.98, hspace=0.0, wspace=0.35)

    for cosmo in ["c001", "c002", "c003"]:

        for i in range(2):
            for j, recon in enumerate(["Prerecon", "RecSym"]):
                c = "r" if recon == "RecSym" else "b"
                axes[j, i].plot([-0.025, 0.025], [-0.025, 0.025], "k--", alpha=0.5)
                axes[j, i].errorbar(
                    results["c000"][recon][:, i + 2], results[cosmo][recon][:, i + 2], linestyle="None", marker="o", color=c, mec="k"
                )
                axes[j, i].set_xlim(-0.025, 0.025)
                axes[j, i].set_ylim(-0.025, 0.025)
                if j == 1:
                    axes[j, i].set_ylabel(
                        f"$b^{{{{c000}}}}_{{{{\\alpha_{{{{||}}}}}}}}\,(\%)$"
                        if i == 0
                        else f"$b^{{{cosmo}}}_{{{{\\alpha_{{{{\\perp}}}}}}}}\,(\%)$",
                        fontsize=14,
                    )
                axes[j, i].set_ylabel(
                    r"$b^{c001}_{\alpha_{||}}$" if i == 0 else r"$b^{c001}_{\alpha_{\perp}}$",
                    fontsize=14,
                    labelpad=-10.0 if i == 0 else 0.0,
                )
                axes[j, i].text(0.30, 0.95, recon, ha="right", va="top", transform=axes[j, i].transAxes, fontsize=14)
        plt.show()

    for cosmo in ["c000", "c001", "c002", "c003"]:

        fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex="all", squeeze=False)
        plt.subplots_adjust(left=0.1, top=0.98, bottom=0.10, right=0.98, hspace=0.0, wspace=0.35)

        for i in range(2):
            for j, recon in enumerate(["Prerecon", "RecSym"]):
                c = "r" if recon == "RecSym" else "b"
                yval = alpha_par_trues[cosmo] if i == 0 else alpha_perp_trues[cosmo]
                m = np.mean(results[cosmo][recon][:, i + 2])
                v = np.std(results[cosmo][recon][:, i + 2]) / np.sqrt(25.0)
                axes[j, i].axhline(y=0.0, color="k", ls="--", alpha=0.5)
                axes[j, i].errorbar(
                    np.arange(25),
                    100.0 * (results[cosmo][recon][:, i + 2]),
                    linestyle="None",
                    marker="o",
                    color=c,
                    mec="k",
                )

                axes[j, i].axhline(y=100.0 * m, color=c, ls="-", lw=1.3)
                axes[j, i].fill_between(np.arange(25), 100.0 * (m - v), 100.0 * (m + v), color=c, alpha=0.3)
                xlim = alpha_par_trues[cosmo] if i == 0 else alpha_perp_trues[cosmo]
                # axes[j, i].set_ylim(xlim - 0.010, xlim + 0.025)
                if j == 1:
                    axes[j, i].set_xlabel(r"$\mathrm{realisation}$", fontsize=14)
                axes[j, i].set_ylabel(
                    f"$b^{{{cosmo}}}_{{{{\\alpha_{{{{||}}}}}}}}\,(\%)$"
                    if i == 0
                    else f"$b^{{{cosmo}}}_{{{{\\alpha_{{{{\\perp}}}}}}}}\,(\%)$",
                    fontsize=14,
                )
                axes[j, i].text(0.30, 0.95, recon, ha="right", va="top", transform=axes[j, i].transAxes, fontsize=14)

        plt.show()
