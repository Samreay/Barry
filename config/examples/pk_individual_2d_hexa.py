import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_cov
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_Beutler2019
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction

# Check to see if including the hexadecapole or higher order multipoles gives tighter constraints on BAO parameters
# when fitting all individual mocks

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [True]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d_quad = PowerSpectrum_Beutler2019(recon=r, isotropic=False, fit_poles=[0, 2])
        d_odd = PowerSpectrum_Beutler2019(recon=r, isotropic=False, fit_poles=[0, 1, 2])
        d_hexa = PowerSpectrum_Beutler2019(recon=r, isotropic=False, fit_poles=[0, 2, 4])
        d_all = PowerSpectrum_Beutler2019(recon=r, isotropic=False, fit_poles=[0, 1, 2, 3, 4])

        # Fix sigma_nl for one of the Beutler models
        model_quad = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.HARTLAP, marg="full"
        )
        model_odd = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 1, 2], correction=Correction.HARTLAP, marg="full"
        )
        model_hexa = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 2, 4], correction=Correction.HARTLAP, marg="full"
        )
        model_all = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 1, 2, 3, 4], correction=Correction.HARTLAP, marg="full"
        )

        fitter.add_model_and_dataset(model_quad, d_quad, name=r"$P_{0}+P_{2}$", linestyle=ls, color=cs[0], realisation="average")
        fitter.add_model_and_dataset(model_odd, d_odd, name=r"$P_{0}+P_{1}+P_{2}$", linestyle=ls, color=cs[1], realisation="average")
        fitter.add_model_and_dataset(model_hexa, d_hexa, name=r"$P_{0}+P_{2}+P_{4}$", linestyle=ls, color=cs[2], realisation="average")
        fitter.add_model_and_dataset(
            model_all, d_all, name=r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$", linestyle=ls, color=cs[3], realisation="average"
        )

        for i in range(997):
            d_quad.set_realisation(i)
            d_odd.set_realisation(i)
            d_hexa.set_realisation(i)
            d_all.set_realisation(i)

            fitter.add_model_and_dataset(model_quad, d_quad, name=r"$P_{0}+P_{2}$", linestyle=ls, color=cs[0], realisation=i)
            fitter.add_model_and_dataset(model_odd, d_odd, name=r"$P_{0}+P_{1}+P_{2}$", linestyle=ls, color=cs[1], realisation=i)
            fitter.add_model_and_dataset(model_hexa, d_hexa, name=r"$P_{0}+P_{2}+P_{4}$", linestyle=ls, color=cs[2], realisation=i)
            fitter.add_model_and_dataset(
                model_all, d_all, name=r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$", linestyle=ls, color=cs[3], realisation=i
            )

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(500)

    if not fitter.should_plot():
        fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        from os import path
        import matplotlib.pyplot as plt
        import logging

        logging.info("Creating plots")
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        res = {}
        bestfits = {}
        if path.exists(pfn + "_alphameans.csv"):
            logging.info("Found alphameans.csv, reading from existing file")

            df = pd.read_csv(pfn + "_alphameans.csv")
            labels = [c.replace("_pk_mean_alpha", "") for c in df.columns if "_pk_mean_alpha" in c]
            for label in labels:
                res[label] = df[
                    [
                        "realisation",
                        label + "_pk_mean_alpha",
                        label + "_pk_std_alpha",
                        label + "_pk_mean_epsilon",
                        label + "_pk_std_epsilon",
                        label + "_pk_chi2",
                    ]
                ].copy()
                res[label].rename(
                    {
                        label + "_pk_mean_alpha": "avg_alpha",
                        label + "_pk_std_alpha": "std_alpha",
                        label + "_pk_mean_epsilon": "avg_epsilon",
                        label + "_pk_std_epsilon": "std_epsilon",
                        label + "_pk_chi2": "chi2",
                    },
                    axis="columns",
                    inplace=True,
                )

            # Print out stats
            if True:
                cols = [c for c in df.columns if "mean" in c]
                df_means = df[cols]
                means = df_means.mean(axis=0)
                rms = df_means.std(axis=0)
                for l, m, r in zip(cols, means, rms):
                    print(f"{l}    ${m:0.3f} \\pm {r:0.3f}$")
        else:

            logging.info("Didn't find alphameans.csv, reading chains")

            doonce = True
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                if extra["realisation"] == "average":
                    continue

                n = extra["name"].split(",")[0]
                if res.get(n) is None:
                    res[n] = []
                if bestfits.get(n) is None:
                    bestfits[n] = []
                i = posterior.argmax()

                model.set_data(data)
                params = model.get_param_dict(chain[i])
                for name, val in params.items():
                    model.set_default(name, val)

                # Ensures we return the window convolved model
                icov_m_w = model.data[0]["icov_m_w"]
                model.data[0]["icov_m_w"][0] = None

                ks = model.data[0]["ks"]
                err = np.sqrt(np.diag(model.data[0]["cov"]))
                mod, mod_odd, polymod, polymod_odd, _ = model.get_model(params, model.data[0], data_name=data[0]["name"])

                if model.marg:
                    mask = data[0]["m_w_mask"]
                    mod_fit, mod_fit_odd = mod[mask], mod_odd[mask]

                    len_poly = len(model.data[0]["ks"]) if model.isotropic else len(model.data[0]["ks"]) * len(model.data[0]["fit_poles"])
                    polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
                    for nn in range(np.shape(polymod)[0]):
                        polymod_fit[nn], polymod_fit_odd[nn] = polymod[nn, mask], polymod_odd[nn, mask]

                    bband = model.get_ML_nuisance(
                        model.data[0]["pk"],
                        mod_fit,
                        mod_fit_odd,
                        polymod_fit,
                        polymod_fit_odd,
                        model.data[0]["icov"],
                        model.data[0]["icov_m_w"],
                    )
                    mod += mod_odd + bband @ (polymod + polymod_odd)
                    mod_fit += mod_fit_odd + bband @ (polymod_fit + polymod_fit_odd)

                    # print(len(model.get_active_params()) + len(bband))
                    # print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
                    new_chi_squared = -2.0 * model.get_chi2_likelihood(
                        model.data[0]["pk"],
                        mod_fit,
                        np.zeros(mod_fit.shape),
                        model.data[0]["icov"],
                        model.data[0]["icov_m_w"],
                        num_mocks=model.data[0]["num_mocks"],
                        num_params=len(model.get_active_params()) + len(bband),
                    )
                    alphas = model.get_alphas(params["alpha"], params["epsilon"])
                    # print(new_chi_squared, len(model.data[0]["pk"]) - len(model.get_active_params()) - len(bband), alphas)

                model.data[0]["icov_m_w"] = icov_m_w

                m, s = weighted_avg_and_cov(chain[:, 0:2], weight, 0)

                if doonce:
                    doonce = False
                    import pandas as pd

                    df = pd.DataFrame(chain[:, 0], columns=["alpha"])
                    nsamp = int((weight / weight.max()).sum())
                    r = []
                    for ii in range(1000):
                        r.append(df.sample(weights=weight, replace=True, n=nsamp).std())
                    print(f"SE of std is {np.std(r)}")

                res[n].append(
                    [
                        m[0],
                        np.sqrt(s[0, 0]),
                        chain[i, 0],
                        m[1],
                        np.sqrt(s[1, 1]),
                        chain[i, 1],
                        s[0, 1],
                        posterior[i],
                        new_chi_squared,
                        -new_chi_squared,
                        extra["realisation"],
                        evidence.max(),
                    ]
                )

                bestfits[n].append(np.concatenate([[extra["realisation"]], chain[i, :], bband]))
            print(res.keys())
            for label in res.keys():
                res[label] = pd.DataFrame(
                    res[label],
                    columns=[
                        "avg_alpha",
                        "std_alpha",
                        "max_alpha",
                        "avg_epsilon",
                        "std_epsilon",
                        "max_epsilon",
                        "corr",
                        "posterior",
                        "chi2",
                        "Dchi2",
                        "realisation",
                        "evidence",
                    ],
                )

            ks = list(res.keys())
            all_ids = pd.concat(tuple([res[l][["realisation"]] for l in ks]))
            counts = all_ids.groupby("realisation").size().reset_index()
            max_count = counts.values[:, 1].max()
            good_ids = counts.loc[counts.values[:, 1] == max_count, ["realisation"]]

            for label, df in res.items():
                res[label] = pd.merge(good_ids, df, how="left", on="realisation")

            df_all = None
            for label, means in res.items():
                d = pd.DataFrame(
                    {
                        "realisation": means["realisation"],
                        f"{label}_pk_mean_alpha": means["avg_alpha"],
                        f"{label}_pk_std_alpha": means["std_alpha"],
                        f"{label}_pk_mean_epsilon": means["avg_epsilon"],
                        f"{label}_pk_std_epsilon": means["std_epsilon"],
                        f"{label}_pk_chi2": means["chi2"],
                    }
                )
                if df_all is None:
                    df_all = d
                else:
                    df_all = pd.merge(df_all, d, how="outer", on="realisation")
            df_all.to_csv(pfn + "_alphameans.csv", index=False, float_format="%0.5f")

            headers = [
                "realisation   alpha   epsilon   sigma_s   beta   sigma_nl_par   sigma_nl_perp   b   a0_1   a0_2   a0_3   a0_4   a0_5   a2_1   a2_2   a2_3   a2_4   a2_5",
                "realisation   alpha   epsilon   sigma_s   beta   sigma_nl_par   sigma_nl_perp   b   a0_1   a0_2   a0_3   a0_4   a0_5   a1_1   a1_2   a1_3   a1_4   a1_5   a2_1   a2_2   a2_3   a2_4   a2_5",
                "realisation   alpha   epsilon   sigma_s   beta   sigma_nl_par   sigma_nl_perp   b   a0_1   a0_2   a0_3   a0_4   a0_5   a2_1   a2_2   a2_3   a2_4   a2_5   a4_1   a4_2   a4_3   a4_4   a4_5",
                "realisation   alpha   epsilon   sigma_s   beta   sigma_nl_par   sigma_nl_perp   b   a0_1   a0_2   a0_3   a0_4   a0_5   a1_1   a1_2   a1_3   a1_4   a1_5   a2_1   a2_2   a2_3   a2_4   a2_5   a3_1   a3_2   a3_3   a3_4   a3_5   a4_1   a4_2   a4_3   a4_4   a4_5",
            ]
            for i, label in enumerate(bestfits.keys()):
                print(headers[i], np.stack(bestfits[label])[0])
                np.savetxt(pfn + "_" + label + "_bestfits.dat", np.transpose(np.stack(bestfits[label]))[:, 0], header=headers[i])

        cols = {
            r"$P_{0}+P_{2}$": cs[0],
            r"$P_{0}+P_{1}+P_{2}$": cs[1],
            r"$P_{0}+P_{2}+P_{4}$": cs[2],
            r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$": cs[3],
        }

        # Bins for means
        bins_alpha = np.linspace(0.94, 1.07, 31)
        lim_alpha = bins_alpha[0], bins_alpha[-1]
        ticks_alpha = [0.97, 1.0, 1.03]

        bins_epsilon = np.linspace(-0.075, 0.075, 31)
        lim_epsilon = bins_epsilon[0], bins_epsilon[-1]
        ticks_epsilon = [-0.05, 0.0, 0.05]

        # Make histogram comparison of the means
        if True:
            fig, axes = plt.subplots(ncols=2, figsize=(5, 3), sharey=True)
            for label, means in res.items():
                c = cols[label.split()[0]]
                axes[0].hist(
                    means["avg_alpha"],
                    bins=bins_alpha,
                    label=" ".join(label.split()[:-1]),
                    histtype="stepfilled",
                    linewidth=2,
                    alpha=0.3,
                    color=c,
                )
                axes[1].hist(means["avg_epsilon"], bins=bins_epsilon, histtype="step", linewidth=1.5, color=c)
            axes[0].set_xlabel(r"$\langle \alpha \rangle$", fontsize=14)
            axes[1].set_xlabel(r"$\langle \epsilon \rangle$", fontsize=14)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])
            axes[0].axvline(1.0, color="k", lw=1, ls="--", alpha=0.6)
            axes[1].axvline(0.0, color="k", lw=1, ls="--", alpha=0.6)
            # axes[0].annotate(r"$\alpha$", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            # axes[1].annotate(r"$\epsilon$", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            # leg1 = axes[0].legend(loc=2, frameon=False)
            # leg2 = axes[1].legend(loc=2, frameon=False)
            # for lh in leg1.legendHandles:
            #    lh.set_alpha(1)
            axes[0].tick_params(axis="y", left=False)
            axes[1].tick_params(axis="y", left=False)
            axes[0].set_xlim(*lim_alpha)
            axes[1].set_xlim(*lim_epsilon)
            plt.subplots_adjust(hspace=0.0)
            fig.savefig(pfn + "_alphahist.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphahist.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Alpha-alpha/epsilon-epsilon comparisons
        if True:
            from scipy.interpolate import interp1d

            # Post-recon
            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
            labels = [r"$P_{0}+P_{2}$", r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "avg_alpha"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(
                            res[label1][k], bins=bins_alpha, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]]
                        )
                        ax.hist(res[label1][k], bins=bins_alpha, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
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
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks_alpha)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim_alpha)
                        ax.set_ylim(*lim_alpha)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(1.0, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            ax.set_yticks(ticks_alpha)
                        if i == 3:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks_alpha)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

            # Pre-recon
            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
            k = "avg_epsilon"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(
                            res[label1][k], bins=bins_epsilon, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]]
                        )
                        ax.hist(res[label1][k], bins=bins_epsilon, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim_epsilon)
                        yval = interp1d(0.5 * (bins_epsilon[:-1] + bins_epsilon[1:]), h, kind="nearest")([0.0])[0]
                        ax.plot([0.0, 0.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 3:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks_epsilon)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim_epsilon)
                        ax.set_ylim(*lim_epsilon)
                        ax.plot([-0.4, 0.4], [-0.4, 0.4], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(0.0, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(0.0, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            ax.set_yticks(ticks_epsilon)
                        if i == 3:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks_epsilon)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_epsiloncomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_epsiloncomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Bins for the alpha errors
        bins_alpha_err = np.linspace(0.008, 0.04, 31)
        lim_alpha_err = bins_alpha_err[0], bins_alpha_err[-1]

        bins_epsilon_err = np.linspace(0.008, 0.04, 31)
        lim_epsilon_err = bins_epsilon_err[0], bins_epsilon_err[-1]

        # Make histogram comparison of the errors
        if True:
            fig, axes = plt.subplots(ncols=2, figsize=(5, 3), sharey=True)
            for label, means in res.items():
                c = cols[label.split()[0]]
                axes[0].hist(
                    means["std_alpha"],
                    bins=bins_alpha,
                    label=" ".join(label.split()[:-1]),
                    histtype="stepfilled",
                    linewidth=2,
                    alpha=0.3,
                    color=c,
                )
                axes[1].hist(means["std_epsilon"], bins=bins_epsilon_err, histtype="step", linewidth=1.5, color=c)
            axes[0].set_xlabel(r"$\sigma_{\alpha}$", fontsize=14)
            axes[1].set_xlabel(r"$\sigma_{\epsilon}$", fontsize=14)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])
            # axes[0].annotate(r"$\alpha$", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            # axes[1].annotate(r"$\epsilon$", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            # leg1 = axes[0].legend(loc=2, frameon=False)
            # leg2 = axes[1].legend(loc=2, frameon=False)
            # for lh in leg1.legendHandles:
            #    lh.set_alpha(1)
            axes[0].tick_params(axis="y", left=False)
            axes[1].tick_params(axis="y", left=False)
            axes[0].set_xlim(*lim_alpha_err)
            axes[1].set_xlim(*lim_epsilon_err)
            plt.subplots_adjust(hspace=0.0)
            fig.savefig(pfn + "_alphaerrhist.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrhist.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Error-error comparison
        if True:
            from scipy.interpolate import interp1d

            # Post-recon
            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
            labels = [r"$P_{0}+P_{2}$", r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "std_alpha"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(
                            res[label1][k],
                            bins=bins_alpha_err,
                            histtype="stepfilled",
                            linewidth=2,
                            alpha=0.3,
                            color=cols[label1.split()[0]],
                        )
                        ax.hist(res[label1][k], bins=bins_alpha_err, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim_alpha_err)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 3:
                            ax.set_xlabel(label2, fontsize=12)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim_alpha_err)
                        ax.set_ylim(*lim_alpha_err)
                        ax.plot([0.0, 1.0], [0.0, 1.0], c="k", lw=1, alpha=0.8, ls=":")

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                        if i == 3:
                            ax.set_xlabel(label2, fontsize=12)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphaerrcomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrcomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
            k = "std_epsilon"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(
                            res[label1][k],
                            bins=bins_epsilon_err,
                            histtype="stepfilled",
                            linewidth=2,
                            alpha=0.3,
                            color=cols[label1.split()[0]],
                        )
                        ax.hist(res[label1][k], bins=bins_epsilon_err, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim_epsilon_err)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 3:
                            ax.set_xlabel(label2, fontsize=12)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim_epsilon_err)
                        ax.set_ylim(*lim_epsilon_err)
                        ax.plot([0.0, 1.0], [0.0, 1.0], c="k", lw=1, alpha=0.8, ls=":")

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                        if i == 3:
                            ax.set_xlabel(label2, fontsize=12)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_epsilonerrcomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_epsilonerrcomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Plot the percentage change in the error relative to , to see how these are correlated
        if True:

            fig, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)
            labels = [r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "avg_alpha"

            for i, label1 in enumerate(labels):

                ax = axes[i]
                print(label1)
                a1 = np.array(res[label1][k] - res[r"$P_{0}+P_{2}$"][k])
                c = np.abs(a1)
                # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                ax.scatter(res[r"$P_{0}+P_{2}$"]["std_alpha"], a1, s=2, c=c, cmap="viridis_r", vmin=0.0, vmax=0.01)
                ax.plot([lim_alpha_err[0], lim_alpha_err[1]], [lim_alpha_err[0], lim_alpha_err[1]], c="k", lw=1, alpha=0.3, ls=":")
                ax.plot([lim_alpha_err[0], lim_alpha_err[1]], [-lim_alpha_err[0], -lim_alpha_err[1]], c="k", lw=1, alpha=0.3, ls=":")

                ax.set_xlim(*lim_alpha_err)
                ax.set_ylim(-0.04, 0.04)
                ax.axhline(y=0.0, c="k", lw=1, alpha=0.8, ls=":")

                if i != 0:
                    # ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                else:
                    ax.set_ylabel(r"$\alpha_{P} - \alpha_{P_{0}+P_{2}}$", fontsize=12)
                ax.set_xlabel(r"$\sigma_{\alpha,P_{0}+P_{2}}$", fontsize=12)
                ax.annotate(label1, (0.02, 0.96), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top")
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphadiff.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphadiff.pdf", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)
            labels = [r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "avg_epsilon"

            for i, label1 in enumerate(labels):

                ax = axes[i]
                print(label1)
                a1 = np.array(res[label1][k] - res[r"$P_{0}+P_{2}$"][k])
                c = np.abs(a1)
                # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                ax.scatter(res[r"$P_{0}+P_{2}$"]["std_epsilon"], a1, s=2, c=c, cmap="viridis_r", vmin=0.0, vmax=0.01)
                ax.plot([lim_epsilon_err[0], lim_epsilon_err[1]], [lim_epsilon_err[0], lim_epsilon_err[1]], c="k", lw=1, alpha=0.3, ls=":")
                ax.plot(
                    [lim_epsilon_err[0], lim_epsilon_err[1]], [-lim_epsilon_err[0], -lim_epsilon_err[1]], c="k", lw=1, alpha=0.3, ls=":"
                )
                ax.set_xlim(*lim_epsilon_err)
                ax.set_ylim(-0.05, 0.05)
                ax.axhline(y=0.0, c="k", lw=1, alpha=0.8, ls=":")

                if i != 0:
                    # ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                else:
                    ax.set_ylabel(r"$\epsilon_{P} - \epsilon_{P_{0}+P_{2}}$", fontsize=12)
                ax.set_xlabel(r"$\sigma_{\epsilon,P_{0}+P_{2}}$", fontsize=12)
                ax.annotate(label1, (0.02, 0.96), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top")
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_epsilondiff.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_epsilondiff.pdf", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)
            labels = [r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "std_alpha"

            for i, label1 in enumerate(labels):

                ax = axes[i]
                print(label1)
                a1 = np.array(res[label1][k] / res[r"$P_{0}+P_{2}$"][k] - 1.0)
                print(np.mean(a1))
                c = np.abs(a1)
                # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                ax.scatter(res[r"$P_{0}+P_{2}$"][k], a1, s=2, c=c, cmap="viridis_r", vmin=0.0, vmax=0.5)
                ax.set_xlim(*lim_alpha_err)
                ax.set_ylim(-1.0, 1.0)
                ax.axhline(y=0.0, c="k", lw=1, alpha=0.8, ls=":")

                if i != 0:
                    # ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                else:
                    ax.set_ylabel(r"$\sigma_{\alpha,P}/\sigma_{\alpha,P_{0}+P_{2}} - 1$", fontsize=12)
                ax.set_xlabel(r"$\sigma_{\alpha,P_{0}+P_{2}}$", fontsize=12)
                ax.annotate(label1, (0.02, 0.96), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top")
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphaerrdiff.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrdiff.pdf", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)
            labels = [r"$P_{0}+P_{1}+P_{2}$", r"$P_{0}+P_{2}+P_{4}$", r"$P_{0}+P_{1}+P_{2}+P_{3}+P_{4}$"]
            k = "std_epsilon"

            for i, label1 in enumerate(labels):

                ax = axes[i]
                print(label1)
                a1 = np.array(res[label1][k] / res[r"$P_{0}+P_{2}$"][k] - 1.0)
                print(np.mean(a1))
                c = np.abs(a1)
                # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                ax.scatter(res[r"$P_{0}+P_{2}$"][k], a1, s=2, c=c, cmap="viridis_r", vmin=0.0, vmax=0.5)
                ax.set_xlim(*lim_epsilon_err)
                ax.set_ylim(-1.0, 1.0)
                ax.axhline(y=0.0, c="k", lw=1, alpha=0.8, ls=":")

                if i != 0:
                    # ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                else:
                    ax.set_ylabel(r"$\sigma_{\epsilon,P}/\sigma_{\epsilon,P_{0}+P_{2}} - 1$", fontsize=12)
                ax.set_xlabel(r"$\sigma_{\epsilon,P_{0}+P_{2}}$", fontsize=12)
                ax.annotate(label1, (0.02, 0.96), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top")
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_epsilonerrdiff.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_epsilonerrdiff.pdf", bbox_inches="tight", dpi=300, transparent=True)
