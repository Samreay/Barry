import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup, weighted_avg_and_std
from barry.models import PowerDing2018, PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True]:  # , False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p, realisation=0)

        ding = PowerDing2018(recon=r)
        beutler = PowerBeutler2017(recon=r)
        sigma_nl = 6.0
        beutler.set_default("sigma_nl", sigma_nl)
        beutler.set_fix_params(["om", "sigma_nl"])

        beutler_extracted = PowerBeutler2017(recon=r, postprocess=p)
        beutler_extracted.set_default("sigma_nl", sigma_nl)
        beutler_extracted.set_fix_params(["om", "sigma_nl"])
        ding_extracted = PowerDing2018(recon=r, postprocess=p)

        for i in range(999):
            d.set_realisation(i)
            de.set_realisation(i)
            fitter.add_model_and_dataset(ding, d, name=f"D18, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler, d, name=f"B17, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(ding_extracted, de, name=f"D18 + Extractor, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler_extracted, de, name=f"B17 + Extractor, mock number {i}", linestyle=ls, color="p", realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(700)
    if not fitter.should_plot():
        fitter.fit(file)

    if fitter.should_plot():
        import matplotlib.pyplot as plt

        import logging

        logging.info("Creating plots")

        res = {}
        rese = {}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            n = extra["name"].split(",")[0]
            if res.get(n) is None:
                res[n] = []
            i = posterior.argmax()
            chi2 = -2 * posterior[i]
            # a, s = weighted_avg_and_std(evidence, weights=weight)
            a, s = weighted_avg_and_std(chain[:, 0], weights=weight)
            res[n].append([a, s, chain[i, 0], posterior[i], chi2, -chi2, extra["realisation"]])

        for label in res.keys():
            res[label] = pd.DataFrame(res[label], columns=["avg", "std", "max", "posterior", "chi2", "Dchi2", "realisation"])

        ks = [l for l in res.keys() if "Smooth" not in l]

        all_ids = pd.concat(tuple([res[l][["realisation"]] for l in ks]))
        counts = all_ids.groupby("realisation").size().reset_index()
        max_count = counts.values[:, 1].max()
        print(counts.values.shape)
        print(all_ids.shape)
        good_ids = counts.loc[counts.values[:, 1] == max_count, ["realisation"]]

        print("Model  ", "Mean mean  ", "Mean std  ", "Std mean")
        for label, df in res.items():
            res[label] = pd.merge(good_ids, df, how="left", on="realisation")
            print(label, np.mean(res[label]["avg"]), np.mean(res[label]["std"]), np.std(res[label]["avg"]))

        # Define colour scheme
        c2 = ["#225465", "#5FA45E"]  # ["#581d7f", "#e05286"]
        c3 = ["#2C455A", "#258E71", "#C1C64D"]  # ["#501b73", "#a73b8f", "#ee8695"]
        c4 = ["#262232", "#116A71", "#48AB75", "#D1E05B"]  # ["#461765", "#7b2a95", "#d54d88", "#f19a9b"]
        c5 = ["#262232", "#1F4D5C", "#0E7A6E", "#5BA561", "#C1C64D"]  # ["#3c1357", "#61208d", "#a73b8f", "#e8638b", "#f4aea3"]
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        if True:
            fig, axes = plt.subplots(nrows=2, figsize=(5, 4), sharex=True, gridspec_kw={"hspace": 0.0})
            pairs = [("B17", "B17 + Extractor"), ("D18", "D18 + Extractor")]

            for pair, ax, index in zip(pairs, axes, [0, 2]):
                combined_df = pd.merge(res[pair[0]], res[pair[1]], on="realisation", suffixes=("_original", "_extracted"))

                data = combined_df[["avg_original", "avg_extracted"]].values
                corr = np.corrcoef(data.T)[1, 0]
                print("corr is ", corr)

                step = 9
                alpha_diff = combined_df["avg_original"] - combined_df["avg_extracted"]
                print(np.mean(alpha_diff), np.std(alpha_diff))
                err = np.sqrt(
                    combined_df["std_original"] ** 2 + combined_df["std_extracted"] ** 2 - 2 * corr * combined_df["std_extracted"] * combined_df["std_original"]
                )
                chi2 = (np.abs(alpha_diff) / err).sum() / alpha_diff.size
                print("chi2 is ", chi2)
                x = combined_df["realisation"][::step]
                ax.errorbar(x, alpha_diff[::step], c=c4[index], yerr=err[::step], fmt="o", elinewidth=0.5, ms=2)
                ax.axhline(0, c="k", lw=1, ls="--")
                ax.set_ylabel(r"$\Delta\alpha$ for " + pair[0], fontsize=14)
                ax.set_ylim(-0.012, 0.012)
            axes[1].set_xlabel("Realisation", fontsize=14)
            plt.savefig(pfn + "_alphadiff.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_alphadiff.png", bbox_inches="tight", dpi=300, transparent=True)

        # Make histogram comparison
        if False:
            cols = {"Beutler": c4[0], "Seo": c4[1], "Ding": c4[2], "Noda": c4[3]}
            fig, axes = plt.subplots(nrows=2, figsize=(5, 6), sharex=True)
            bins = np.linspace(0.73, 1.15, 31)
            for label, means in res.items():
                if "Prerecon" in label:
                    ax = axes[0]
                else:
                    ax = axes[1]
                ax.hist(means, bins=bins, label=label, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label.split()[0]])
                ax.hist(means, bins=bins, histtype="step", linewidth=1.5, color=cols[label.split()[0]])
            axes[1].set_xlabel(r"$\langle \alpha \rangle$", fontsize=14)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])
            axes[0].axvline(0.9982, color="k", lw=1, ls="--", alpha=0.6)
            axes[1].axvline(0.9982, color="k", lw=1, ls="--", alpha=0.6)
            leg1 = axes[0].legend(loc=2, frameon=False)
            leg2 = axes[1].legend(loc=2, frameon=False)
            for lh in leg1.legendHandles + leg2.legendHandles:
                lh.set_alpha(1)
            axes[0].tick_params(axis="y", left=False)
            axes[1].tick_params(axis="y", left=False)
            plt.subplots_adjust(hspace=0.0)
            fig.savefig(pfn + "_alphahist.png", bbox_inches="tight", dpi=300, transparent=True)

        from matplotlib.colors import to_rgb, to_hex

        def blend_hex(hex1, hex2):
            a = np.array(to_rgb(hex1))
            b = np.array(to_rgb(hex2))
            return to_hex(0.5 * (a + b))

        # Alpha-alpha comparison
        if False:
            from scipy.interpolate import interp1d

            bins_both = np.linspace(0.92, 1.08, 31)
            bins = np.linspace(0.95, 1.06, 31)
            ticks = [0.97, 1.0, 1.03]
            lim = bins[0], bins[-1]
            lim_both = bins_both[0], bins_both[-1]

            cols = {"B17": c4[0], "B17 + Extractor": c4[0], "D18": c4[2], "D18 + Extractor": c4[2]}
            fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
            labels = ["B17", "B17 + Extractor", "D18", "D18 + Extractor"]
            k = "avg"

            # labels = ["Beutler Prerecon", "Seo Prerecon", "Ding Prerecon", "Noda Prerecon"]
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(res[label1][k], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1])
                        ax.hist(res[label1][k], bins=bins, histtype="step", linewidth=1.5, color=cols[label1])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(*lim)
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([1.0])[0]
                        ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 3:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][k])
                        a2 = np.array(res[label1][k])
                        c = np.abs(a1 - a2)
                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.0002, vmax=0.01)
                        ax.set_xlim(lim[0], lim[1])
                        ax.set_ylim(*lim)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(0.9982, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(0.9982, color="k", lw=1, ls="--", alpha=0.4)

                        if "Extractor" in label1 or "Extractor" in label2:
                            if "Extractor" in label1 and "Extractor" in label2:
                                print("DARK")
                                ax.set_facecolor("#d9d9d9")
                            else:
                                print("LIGHT")
                                ax.set_facecolor("#f0f0f0")

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                            ax.set_yticks(ticks)
                        if i == 3:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300)
