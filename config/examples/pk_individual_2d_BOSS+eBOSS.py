import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import DynestySampler, Optimiser
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_cov
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12, PowerSpectrum_eBOSS_LRGpCMASS
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction


# Run an optimisation on each of the post-recon SDSS DR12 mocks. Then compare to the pre-recon mocks
# to compute the cross-correlation between BAO parameters and pre-recon measurements

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()

    # sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    sampler = Optimiser(temp_dir=dir_name, tol=1.0e-4)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    datasets = [
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1, galactic_cap="both", recon="iso", isotropic=False, fit_poles=[0, 2], min_k=0.02, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2, galactic_cap="both", recon="iso", isotropic=False, fit_poles=[0, 2], min_k=0.02, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_eBOSS_LRGpCMASS(
            galactic_cap="both", recon="iso", isotropic=False, fit_poles=[0, 2, 4], min_k=0.02, max_k=0.30, num_mocks=999
        ),
    ]

    pre_recon_datasets = [
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1, galactic_cap="both", recon=None, isotropic=False, fit_poles=[0, 2, 4], min_k=0.0, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2, galactic_cap="both", recon=None, isotropic=False, fit_poles=[0, 2, 4], min_k=0.0, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_eBOSS_LRGpCMASS(
            galactic_cap="both", recon=None, isotropic=False, fit_poles=[0, 2, 4], min_k=0.0, max_k=0.30, num_mocks=999
        ),
    ]

    # Standard Beutler Model
    model = PowerBeutler2017(
        recon="iso", isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.HARTLAP, marg="full", n_data=2
    )
    model4 = PowerBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om"],
        poly_poles=[0, 2, 4],
        correction=Correction.HARTLAP,
        marg="full",
        n_data=2,
    )

    for d in datasets:
        for i in range(999):
            d.set_realisation(i)
            if 4 in d.fit_poles:
                fitter.add_model_and_dataset(model4, d, name=d.name + " " + str(i), realisation=i)
            else:
                fitter.add_model_and_dataset(model, d, name=d.name + " " + str(i), realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(100)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        from os import path
        import matplotlib.pyplot as plt
        import logging

        logging.info("Creating plots")
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        res = []
        if path.exists(pfn + "_alphameans.pkl"):
            logging.info("Found alphameans.pkl, reading from existing file")

            res = pd.read_pickle(pfn + "_alphameans.pkl")
        else:

            logging.info("Didn't find alphameans.pkl, reading chains")

            counter = 0
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():

                n = "_".join(extra["name"].split()[:-2])
                print(n)

                model.set_data(data)
                params = model.get_param_dict(chain[0])
                for name, val in params.items():
                    model.set_default(name, val)

                alphas = model.get_alphas(params["alpha"], params["epsilon"])
                new_chi_squared, dof, bband, mods, smooths = model.plot(params, display=False)

                dataset = int(counter / 999)
                print(dataset, counter % 999)
                pre_recon_data = pre_recon_datasets[dataset].set_realisation(counter % 999)
                pre_recon_data = pre_recon_data.get_data()

                print(params["alpha"], params["epsilon"], alphas, new_chi_squared)

                res.append(
                    [
                        n,
                        counter % 999,
                        new_chi_squared,
                        params["alpha"],
                        1.0 + params["epsilon"],
                        alphas[0],
                        alphas[1],
                        model.data[0]["ks"],
                        model.data[0]["pk0"],
                        model.data[0]["pk2"],
                        model.data[0]["pk4"],
                        pre_recon_data[0]["ks"],
                        pre_recon_data[0]["pk0"],
                        pre_recon_data[0]["pk2"],
                        pre_recon_data[0]["pk4"],
                    ]
                )

                counter += 1

            res = pd.DataFrame(
                res,
                columns=[
                    "patch",
                    "realisation",
                    "chi_squared",
                    "alpha",
                    "epsilon",
                    "alpha_par",
                    "alpha_perp",
                    "recon_ks",
                    "recon_pk0",
                    "recon_pk2",
                    "recon_pk4",
                    "ks",
                    "pk0",
                    "pk2",
                    "pk4",
                ],
            )

            res.to_pickle(pfn + "_alphameans.pkl")

        # Bins for means
        bins_alpha = np.linspace(0.85, 1.15, 31)
        lim_alpha = bins_alpha[0], bins_alpha[-1]
        ticks_alpha = [0.90, 1.0, 1.10]

        bins_epsilon = np.linspace(0.85, 1.15, 31)
        lim_epsilon = bins_epsilon[0], bins_epsilon[-1]
        ticks_epsilon = [0.90, 1.0, 1.10]

        # Alpha-alpha/epsilon-epsilon comparisons
        if False:
            from scipy.interpolate import interp1d

            # Plots for each patch
            for label in [
                "BOSS_DR12_Pk_ngc_z1",
                "BOSS_DR12_Pk_sgc_z1",
                "BOSS_DR12_Pk_ngc_z2",
                "BOSS_DR12_Pk_sgc_z2",
                "eBOSS_DR16_LRGpCMASS_Pk_NGC",
                "eBOSS_DR16_LRGpCMASS_Pk_SGC",
            ]:
                df = res.drop(res[res["patch"] != label].index)
                fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
                labels = ["alpha", "epsilon", "alpha_perp", "alpha_par"]
                str_labels = [r"$\alpha$", r"1 + $\epsilon$", r"$\alpha_{\perp}$", r"$\alpha_{||}$"]

                for i, (label1, str_label1) in enumerate(zip(labels, str_labels)):
                    for j, (label2, str_label2) in enumerate(zip(labels, str_labels)):
                        ax = axes[i, j]
                        if i < j:
                            ax.axis("off")
                            continue
                        elif i == j:
                            h, _, _ = ax.hist(df[label1], bins=bins_alpha, histtype="stepfilled", linewidth=2, alpha=0.3, color="#262232")
                            ax.hist(df[label1], bins=bins_alpha, histtype="step", linewidth=1.5, color="#262232")
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                            ax.set_xlim(*lim_alpha)
                            yval = interp1d(0.5 * (bins_alpha[:-1] + bins_alpha[1:]), h, kind="nearest")([1.0])[0]
                            ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                            ax.spines["right"].set_visible(False)
                            ax.spines["top"].set_visible(False)
                            if j == 0:
                                ax.spines["left"].set_visible(False)
                            if j == 3:
                                ax.set_xlabel(str_label2, fontsize=12)
                                ax.set_xticks(ticks_alpha)
                        else:
                            a1 = np.array(res[label2])
                            a2 = np.array(res[label1])
                            c = np.fabs(1.0 - a2 ** (1.0 / 3.0) * a1 ** (2.0 / 3.0))

                            ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                            ax.set_xlim(*lim_alpha)
                            ax.set_ylim(*lim_alpha)
                            ax.plot(np.linspace(0.8, 1.2, 100), 1.0 / np.linspace(0.8, 1.2, 100) ** 2, c="k", lw=1, alpha=0.8, ls=":")
                            ax.axvline(1.0, color="k", lw=1, ls="--", alpha=0.4)
                            ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4)

                            if j != 0:
                                ax.set_yticklabels([])
                                ax.tick_params(axis="y", left=False)
                            else:
                                ax.set_ylabel(str_label1, fontsize=12)
                                ax.set_yticks(ticks_alpha)
                            ax.set_xlabel(str_label2, fontsize=12)
                            ax.set_xticks(ticks_alpha)
                plt.subplots_adjust(hspace=0.0, wspace=0)
                fig.savefig(pfn + "_" + label + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
                fig.savefig(pfn + "_" + label + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Plot correlation coefficients
        if False:
            from scipy.interpolate import interp1d

            # Plots for each patch
            for label in [
                "BOSS_DR12_Pk_ngc_z1",
                "BOSS_DR12_Pk_sgc_z1",
                "BOSS_DR12_Pk_ngc_z2",
                "BOSS_DR12_Pk_sgc_z2",
                "eBOSS_DR16_LRGpCMASS_Pk_NGC",
                "eBOSS_DR16_LRGpCMASS_Pk_SGC",
            ]:
                df = res.drop(res[res["patch"] != label].index).reset_index()
                print(df)

                data = np.array(
                    [
                        np.concatenate(
                            [
                                [df["alpha_perp"][i]],
                                [df["alpha_par"][i]],
                                df["pk0"].to_numpy()[i],
                                df["pk2"].to_numpy()[i],
                                df["pk4"].to_numpy()[i],
                            ]
                        )
                        for i in range(len(df))
                    ]
                ).T

                cov = np.cov(data)
                corr = np.corrcoef(data)

                print(np.sqrt(np.diag(cov[:2, :2])), corr[0, :], corr[1, :])

                labels = ["alpha_perp", "alpha_par"]
                str_labels = [r"$\alpha_{\perp}$", r"$\alpha_{||}$"]

                fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)
                for i, (label1, str_label1) in enumerate(zip(labels, str_labels)):

                    ax = axes[i]
                    print(label1)

                    ax.plot(df["ks"][0], corr[i, 2 : len(df["ks"][0]) + 2], color="r", ls="-")
                    ax.plot(df["ks"][0], corr[i, len(df["ks"][0]) + 2 : 2 * len(df["ks"][0]) + 2], color="b", ls="-")
                    ax.plot(df["ks"][0], corr[i, 2 * len(df["ks"][0]) + 2 :], color="g", ls="-")
                    ax.axhline(y=0.0, c="k", lw=1, alpha=0.3, ls=":")

                    ax.set_xlim(0.0, 0.3)
                    ax.set_ylim(-0.25, 0.25)

                    if i != 0:
                        ax.tick_params(axis="y", left=False)
                    else:
                        ax.set_ylabel(r"$\rho_{\alpha,P}$", fontsize=12)
                    ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=12)
                    ax.annotate(str_label1, (0.02, 0.96), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top")
                plt.subplots_adjust(hspace=0.0, wspace=0)
                fig.savefig(pfn + "_" + label + "_alphacorr.png", bbox_inches="tight", dpi=300, transparent=True)
                fig.savefig(pfn + "_" + label + "_alphacorr.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Evaluate the covariance matrix
        if True:

            data = []
            for label in [
                "BOSS_DR12_Pk_ngc_z1",
                "BOSS_DR12_Pk_sgc_z1",
                "BOSS_DR12_Pk_ngc_z2",
                "BOSS_DR12_Pk_sgc_z2",
            ]:
                df = res.drop(res[res["patch"] != label].index).reset_index()
                data.append(
                    np.array(
                        [
                            np.concatenate(
                                [
                                    df["pk0"].to_numpy()[i],
                                    df["pk2"].to_numpy()[i],
                                    df["pk4"].to_numpy()[i],
                                    [df["alpha_perp"][i]],
                                    [df["alpha_par"][i]],
                                ]
                            )
                            for i in range(len(df))
                        ]
                    ).T
                )
            data = np.concatenate(data)

            cov = np.cov(data)
            np.savetxt(pfn + "Cov_matrix_BOSS_DR12_NGC_SGC_z1_z2_kmax_0.3_deltak_0p01_Mono+Quad+Hex_postreconBAO.txt", cov)

            data = []
            for label in [
                "eBOSS_DR16_LRGpCMASS_Pk_NGC",
            ]:
                df = res.drop(res[res["patch"] != label].index).reset_index()
                data.append(
                    np.array(
                        [
                            np.concatenate(
                                [
                                    df["pk0"].to_numpy()[i],
                                    df["pk2"].to_numpy()[i],
                                    df["pk4"].to_numpy()[i],
                                    [df["alpha_perp"][i]],
                                    [df["alpha_par"][i]],
                                ]
                            )
                            for i in range(len(df))
                        ]
                    ).T
                )
            data = np.concatenate(data)

            cov = np.cov(data)
            np.savetxt(pfn + "Cov_matrix_eBOSS_DR16_LRGpCMASS_NGC_kmax_0.3_deltak_0p01_Mono+Quad+Hex_postreconBAO.txt", cov)

            data = []
            for label in [
                "eBOSS_DR16_LRGpCMASS_Pk_SGC",
            ]:
                df = res.drop(res[res["patch"] != label].index).reset_index()
                data.append(
                    np.array(
                        [
                            np.concatenate(
                                [
                                    df["pk0"].to_numpy()[i],
                                    df["pk2"].to_numpy()[i],
                                    df["pk4"].to_numpy()[i],
                                    [df["alpha_perp"][i]],
                                    [df["alpha_par"][i]],
                                ]
                            )
                            for i in range(len(df))
                        ]
                    ).T
                )
            data = np.concatenate(data)

            cov = np.cov(data)
            np.savetxt(pfn + "Cov_matrix_eBOSS_DR16_LRGpCMASS_SGC_kmax_0.3_deltak_0p01_Mono+Quad+Hex_postreconBAO.txt", cov)
