import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.fitter import Fitter
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, save_dims=2, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p, realisation=0)

        beutler_not_fixed = PowerBeutler2017(recon=r)
        beutler = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        beutler.set_default("sigma_nl", sigma_nl)
        beutler.set_fix_params(["om", "sigma_nl"])

        seo = PowerSeo2016(recon=r)
        ding = PowerDing2018(recon=r)
        noda = PowerNoda2019(recon=r, postprocess=p)

        for i in range(999):
            d.set_realisation(i)
            de.set_realisation(i)

            fitter.add_model_and_dataset(beutler_not_fixed, d, name=f"Beutler 2017 {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(seo, d, name=f"Seo 2016 {t}, mock number {i}", linestyle=ls, color="r", realisation=i)
            fitter.add_model_and_dataset(ding, d, name=f"Ding 2018 {t}, mock number {i}", linestyle=ls, color="lb", realisation=i)
            fitter.add_model_and_dataset(noda, de, name=f"Noda 2019 {t}, mock number {i}", linestyle=ls, color="o", realisation=i)

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
        if path.exists(pfn + "_alphameans.csv"):
            logging.info("Found alphameans.csv, reading from existing file")

            df = pd.read_csv(pfn + "_alphameans.csv")
            labels = [c.replace("_pk_mean", "") for c in df.columns if "_pk_mean" in c]
            for label in labels:
                res[label] = df[[label + "_pk_mean", label + "_pk_std", "realisation", label + "_pk_evidence"]].copy()
                res[label].rename({label + "_pk_mean": "avg", label + "_pk_std": "std", label + "_pk_evidence": "evidence"}, axis="columns", inplace=True)

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
                n = extra["name"].split(",")[0]
                if res.get(n) is None:
                    res[n] = []
                i = posterior.argmax()
                chi2 = -2 * posterior[i]
                m, s = weighted_avg_and_std(chain[:, 0], weight)

                if doonce:
                    doonce = False
                    import pandas as pd

                    df = pd.DataFrame(chain[:, 0], columns=["alpha"])
                    nsamp = int((weight / weight.max()).sum())
                    r = []
                    for ii in range(1000):
                        r.append(df.sample(weights=weight, replace=True, n=nsamp).std())
                    print(f"SE of std is {np.std(r)}")

                res[n].append([m, s, chain[i, 0], posterior[i], chi2, -chi2, extra["realisation"], evidence.max()])
            for label in res.keys():
                res[label] = pd.DataFrame(res[label], columns=["avg", "std", "max", "posterior", "chi2", "Dchi2", "realisation", "evidence"])

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
                        f"{label}_pk_mean": means["avg"],
                        f"{label}_pk_std": means["std"],
                        f"{label}_pk_evidence": means["evidence"],
                    }
                )
                if df_all is None:
                    df_all = d
                else:
                    df_all = pd.merge(df_all, d, how="outer", on="realisation")
            df_all.to_csv(pfn + "_alphameans.csv", index=False, float_format="%0.5f")

        # Define colour scheme
        c2 = ["#225465", "#5FA45E"]  # ["#581d7f", "#e05286"]
        c3 = ["#2C455A", "#258E71", "#C1C64D"]  # ["#501b73", "#a73b8f", "#ee8695"]
        c4 = ["#262232", "#116A71", "#48AB75", "#D1E05B"]  # ["#461765", "#7b2a95", "#d54d88", "#f19a9b"]
        c5 = ["#262232", "#1F4D5C", "#0E7A6E", "#5BA561", "#C1C64D"]  # ["#3c1357", "#61208d", "#a73b8f", "#e8638b", "#f4aea3"]
        cols = {"Beutler": c4[0], "Seo": c4[1], "Ding": c4[2], "Noda": c4[3]}

        # Bins for means
        bins_both = np.linspace(0.91, 1.08, 31)
        bins = np.linspace(0.95, 1.06, 31)
        ticks = [0.97, 1.0, 1.03]
        lim = bins[0], bins[-1]
        lim_both = bins_both[0], bins_both[-1]

        # Make histogram comparison of the means
        if False:
            fig, axes = plt.subplots(nrows=2, figsize=(5, 4), sharex=True)
            for label, means in res.items():
                if "Prerecon" in label:
                    ax = axes[0]
                else:
                    ax = axes[1]
                c = cols[label.split()[0]]
                ax.hist(means["avg"], bins=bins_both, label=" ".join(label.split()[:-1]), histtype="stepfilled", linewidth=2, alpha=0.3, color=c)
                ax.hist(means["avg"], bins=bins_both, histtype="step", linewidth=1.5, color=cols[label.split()[0]])
            axes[1].set_xlabel(r"$\langle \alpha \rangle$", fontsize=14)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])
            axes[0].axvline(0.9982, color="k", lw=1, ls="--", alpha=0.6)
            axes[1].axvline(0.9982, color="k", lw=1, ls="--", alpha=0.6)
            axes[0].annotate("Prerecon", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            axes[1].annotate("Recon", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            leg1 = axes[0].legend(loc=2, frameon=False)
            # leg2 = axes[1].legend(loc=2, frameon=False)
            for lh in leg1.legendHandles:
                lh.set_alpha(1)
            axes[0].tick_params(axis="y", left=False)
            axes[1].tick_params(axis="y", left=False)
            axes[0].set_xlim(*lim_both)
            axes[1].set_xlim(*lim_both)
            plt.subplots_adjust(hspace=0.0)
            fig.savefig(pfn + "_alphahist.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphahist.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Alpha-alpha comparison
        if False:
            from scipy.interpolate import interp1d

            # Post-recon
            cols = {"Beutler": c4[0], "Seo": c4[1], "Ding": c4[2], "Noda": c4[3]}
            fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True)
            labels = ["Beutler 2017 Recon", "Beutler 2017 Fixed $\\Sigma_{nl}$ Recon", "Seo 2016 Recon", "Ding 2018 Recon", "Noda 2019 Recon"]
            k = "avg"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(res[label1][k], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                        ax.hist(res[label1][k], bins=bins, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim)
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([1.0])[0]
                        ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim)
                        ax.set_ylim(*lim)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(0.9982, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(0.9982, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            ax.set_yticks(ticks)
                        if i == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

            # Pre-recon
            fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True)
            labels = ["Beutler 2017 Prerecon", "Beutler 2017 Fixed $\\Sigma_{nl}$ Prerecon", "Seo 2016 Prerecon", "Ding 2018 Prerecon", "Noda 2019 Prerecon"]
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(res[label1][k], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                        ax.hist(res[label1][k], bins=bins, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim)
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([1.0])[0]
                        ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter([a1[argbad]], [a2[argbad]], s=2, c='r', zorder=10)

                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0005, vmax=0.02)
                        ax.set_xlim(*lim)
                        ax.set_ylim(*lim)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(0.9982, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(0.9982, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            ax.set_yticks(ticks)
                        if i == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            ax.set_xticks(ticks)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp_prerecon.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp_prerecon.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Bins for the alpha errors
        bins_both = np.linspace(0.005, 0.04, 31)
        lim_both = bins_both[0], bins_both[-1]

        # Make histogram comparison of the errors
        if False:
            fig, axes = plt.subplots(nrows=2, figsize=(5, 4), sharex=True)
            for label, means in res.items():
                if "Prerecon" in label:
                    ax = axes[0]
                else:
                    ax = axes[1]
                c = cols[label.split()[0]]
                ax.hist(means["std"], bins=bins_both, label=" ".join(label.split()[:-1]), histtype="stepfilled", linewidth=2, alpha=0.3, color=c)
                ax.hist(means["std"], bins=bins_both, histtype="step", linewidth=1.5, color=cols[label.split()[0]])
            axes[1].set_xlabel(r"$\sigma_{\alpha}$", fontsize=14)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])
            axes[0].annotate("Prerecon", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            axes[1].annotate("Recon", (0.98, 0.96), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top")
            # leg1 = axes[0].legend(loc=1, frameon=False)
            leg2 = axes[1].legend(loc=4, frameon=False)
            for lh in leg2.legendHandles:
                lh.set_alpha(1)
            axes[0].tick_params(axis="y", left=False)
            axes[1].tick_params(axis="y", left=False)
            axes[0].set_xlim(*lim_both)
            axes[1].set_xlim(*lim_both)
            plt.subplots_adjust(hspace=0.0)
            fig.savefig(pfn + "_alphaerrhist.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrhist.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Error-error comparison
        if False:
            from scipy.interpolate import interp1d

            # Post-recon
            bins = np.linspace(0.006, 0.028, 31)
            bins2 = np.linspace(0.006, 0.018, 31)
            ticks = [0.010, 0.017, 0.024]
            ticks2 = [0.008, 0.012, 0.016]
            lim = bins[0], bins[-1]
            lim2 = bins2[0], bins2[-1]

            cols = {"Beutler": c4[0], "Seo": c4[1], "Ding": c4[2], "Noda": c4[3]}
            fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=False)
            labels = ["Beutler 2017 Recon", "Beutler 2017 Fixed $\\Sigma_{nl}$ Recon", "Seo 2016 Recon", "Ding 2018 Recon", "Noda 2019 Recon"]
            k = "std"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        if label1 == "Beutler 2017 Recon":
                            h, _, _ = ax.hist(res[label1][k], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                            ax.hist(res[label1][k], bins=bins, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        else:
                            h, _, _ = ax.hist(res[label1][k], bins=bins2, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                            ax.hist(res[label1][k], bins=bins2, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        if label1 == "Beutler 2017 Recon":
                            ax.set_xlim(*lim)
                        else:
                            ax.set_xlim(*lim2)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            if label2 == "Beutler 2017 Recon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                        else:
                            if label2 == "Beutler 2017 Recon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                            ax.set_xticklabels([])
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter(a1, a2, s=2, c=means["avg"], cmap="viridis_r", vmin=0.92, vmax=1.08)
                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0001, vmax=0.006)
                        if label1 == "Beutler 2017 Recon":
                            ax.set_ylim(*lim)
                        else:
                            ax.set_ylim(*lim2)
                        if label2 == "Beutler 2017 Recon":
                            ax.set_xlim(*lim)
                        else:
                            ax.set_xlim(*lim2)
                        ax.plot([0.0, 1.0], [0.0, 1.0], c="k", lw=1, alpha=0.8, ls=":")

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            if label1 == "Beutler 2017 Recon":
                                ax.set_yticks(ticks)
                            else:
                                ax.set_yticks(ticks2)
                        if i == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            if label2 == "Beutler 2017 Recon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                        else:
                            if label2 == "Beutler 2017 Recon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                            ax.set_xticklabels([])
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphaerrcomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrcomp.pdf", bbox_inches="tight", dpi=300, transparent=True)

            # Pre-recon
            bins = np.linspace(0.005, 0.055, 31)
            bins2 = np.linspace(0.005, 0.035, 31)
            ticks = [0.01, 0.03, 0.05]
            ticks2 = [0.01, 0.02, 0.03]
            lim = bins[0], bins[-1]
            lim2 = bins2[0], bins2[-1]

            fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=False)
            labels = ["Beutler 2017 Prerecon", "Beutler 2017 Fixed $\\Sigma_{nl}$ Prerecon", "Seo 2016 Prerecon", "Ding 2018 Prerecon", "Noda 2019 Prerecon"]
            k = "std"
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        if label1 == "Beutler 2017 Prerecon":
                            h, _, _ = ax.hist(res[label1][k], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                            ax.hist(res[label1][k], bins=bins, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        else:
                            h, _, _ = ax.hist(res[label1][k], bins=bins2, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1.split()[0]])
                            ax.hist(res[label1][k], bins=bins2, histtype="step", linewidth=1.5, color=cols[label1.split()[0]])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        if label1 == "Beutler 2017 Prerecon":
                            ax.set_xlim(*lim)
                        else:
                            ax.set_xlim(*lim2)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            if label2 == "Beutler 2017 Prerecon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                        else:
                            if label2 == "Beutler 2017 Prerecon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                            ax.set_xticklabels([])
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        # ax.scatter(a1, a2, s=2, c=means["avg"], cmap="viridis_r", vmin=0.92, vmax=1.08)
                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0001, vmax=0.012)
                        if label1 == "Beutler 2017 Prerecon":
                            ax.set_ylim(*lim)
                        else:
                            ax.set_ylim(*lim2)
                        if label2 == "Beutler 2017 Prerecon":
                            ax.set_xlim(*lim)
                        else:
                            ax.set_xlim(*lim2)
                        ax.plot([0.0, 1.0], [0.0, 1.0], c="k", lw=1, alpha=0.8, ls=":")

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(" ".join(label1.split()[:-1]), fontsize=12)
                            if label1 == "Beutler 2017 Prerecon":
                                ax.set_yticks(ticks)
                            else:
                                ax.set_yticks(ticks2)
                        if i == 4:
                            ax.set_xlabel(" ".join(label2.split()[:-1]), fontsize=12)
                            if label2 == "Beutler 2017 Prerecon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                        else:
                            if label2 == "Beutler 2017 Prerecon":
                                ax.set_xticks(ticks)
                            else:
                                ax.set_xticks(ticks2)
                            ax.set_xticklabels([])
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphaerrcomp_prerecon.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphaerrcomp_prerecon.pdf", bbox_inches="tight", dpi=300, transparent=True)

        # Plot the error as a function of the mean, to see how these are correlated
        if False:

            import matplotlib.gridspec as gridspec

            # Post-recon
            fig, axes = plt.subplots(figsize=(6, 8), nrows=1, sharex=True)
            inner = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=axes, hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\sigma_{\alpha}$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for label, means in res.items():
                if not "Recon" in label:
                    continue
                ax = fig.add_subplot(inner[counter])
                c = np.abs(means["std"] - np.mean(means["std"]))
                ax.scatter(means["avg"], means["std"], s=2, c=c, cmap="viridis_r", vmin=-0.0001, vmax=0.01, label=label)
                ax.axhline(y=np.mean(means["std"]), c="k", lw=1, alpha=0.8, ls=":")
                ax.set_xlim(0.96, 1.04)
                if label == "Beutler 2017 Recon":
                    ax.set_ylim(0.006, 0.028)
                else:
                    ax.set_ylim(0.006, 0.018)
                ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax.legend(loc="upper right", fontsize=9)
                counter += 1
            ax.tick_params(axis="x", which="both", labelcolor="k", bottom=True, labelbottom=True)
            plt.savefig(pfn + "_alphameanerr.pdf")

            # Pre-recon
            fig, axes = plt.subplots(figsize=(6, 8), nrows=1, sharex=True)
            inner = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=axes, hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\sigma_{\alpha}$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for label, means in res.items():
                if not "Prerecon" in label:
                    continue
                ax = fig.add_subplot(inner[counter])
                c = np.abs(means["std"] - np.mean(means["std"]))
                ax.scatter(means["avg"], means["std"], s=2, c=c, cmap="viridis_r", vmin=-0.0001, vmax=0.01, label=label)
                ax.axhline(y=np.mean(means["std"]), c="k", lw=1, alpha=0.8, ls=":")
                ax.set_xlim(0.91, 1.09)
                if label == "Beutler 2017 Prerecon":
                    ax.set_ylim(0.006, 0.055)
                else:
                    ax.set_ylim(0.006, 0.035)
                ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax.legend(loc="upper right", fontsize=9)
                counter += 1
            ax.tick_params(axis="x", which="both", labelcolor="k", bottom=True, labelbottom=True)
            plt.savefig(pfn + "_alphameanerr_prerecon.pdf")
