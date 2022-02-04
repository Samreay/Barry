import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.config import setup
from barry.models import PowerDing2018, CorrDing2018
from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC, PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd


# Check correlation between pk and xi results using recon Ding
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, save_dims=2, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=400)

    for r in [True]:
        t = "Recon" if r else "Prerecon"

        d_pk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        d_xi = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=r, realisation=0)

        ding_pk = PowerDing2018(recon=r)
        ding_pk_smooth = PowerDing2018(recon=r, smooth=True)

        ding_xi = CorrDing2018(recon=r)
        ding_xi_smooth = CorrDing2018(recon=r, smooth=True)

        for i in range(999):
            d_pk.set_realisation(i)
            d_xi.set_realisation(i)

            fitter.add_model_and_dataset(ding_pk, d_pk, name=f"Ding 2018 $P(k)$, mock number {i}", linestyle="-", color="p", realisation=i)
            fitter.add_model_and_dataset(ding_pk_smooth, d_pk, name=f"Ding 2018 $P(k)$ Smooth, mock number {i}", linestyle="-", color="p", realisation=i)

            fitter.add_model_and_dataset(ding_xi, d_xi, name=f"Ding 2018 $\\xi(s)$, mock number {i}", linestyle=":", color="p", realisation=i)
            fitter.add_model_and_dataset(ding_xi_smooth, d_xi, name=f"Ding 2018 $\\xi(s)$ Smooth, mock number {i}", linestyle=":", color="p", realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(400)
    if not fitter.should_plot():
        fitter.fit(file)

    if fitter.should_plot():
        import matplotlib.pyplot as plt

        import logging

        logging.info("Creating plots")

        res = {}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            n = extra["name"].split(",")[0]
            realisation = extra["realisation"]
            if res.get(n) is None:
                res[n] = []
            i = posterior.argmax()
            chi2 = -2 * posterior[i]
            # res[n].append([np.average(evidence, weights=weight), np.std(chain[:, 0]), chain[i, 0], posterior[i], chi2, -chi2, extra["realisation"]])
            res[n].append([np.average(chain[:, 0], weights=weight), np.std(chain[:, 0]), chain[i, 0], posterior[i], chi2, -chi2, extra["realisation"]])
        for label in res.keys():
            res[label] = pd.DataFrame(res[label], columns=["avg", "std", "max", "posterior", "chi2", "Dchi2", "realisation"])

        for label, df in res.items():
            if "Smooth" not in label:
                l2 = label + " Smooth"
                res[label] = pd.merge(df, res[l2], how="inner", on="realisation", suffixes=("", "_"))
                res[label]["Dchi2"] += res[label]["chi2_"]
                print(label, res[label]["Dchi2"].max(), res[label]["Dchi2"].mean())

        ks = [l for l in res.keys() if "Smooth" not in l]
        all_ids = pd.concat(tuple([res[l][["realisation"]] for l in ks]))
        counts = all_ids.groupby("realisation").size().reset_index()
        max_count = counts.values[:, 1].max()
        good_ids = counts.loc[counts.values[:, 1] == max_count, ["realisation"]]

        # Define colour scheme
        c2 = ["#225465", "#5FA45E"]  # ["#581d7f", "#e05286"]
        c3 = ["#2C455A", "#258E71", "#C1C64D"]  # ["#501b73", "#a73b8f", "#ee8695"]
        c4 = ["#262232", "#116A71", "#48AB75", "#D1E05B"]  # ["#461765", "#7b2a95", "#d54d88", "#f19a9b"]
        c5 = ["#262232", "#1F4D5C", "#0E7A6E", "#5BA561", "#C1C64D"]  # ["#3c1357", "#61208d", "#a73b8f", "#e8638b", "#f4aea3"]

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        bins_both = np.linspace(0.92, 1.08, 31)
        bins = np.linspace(0.95, 1.06, 31)
        ticks = [0.97, 1.0, 1.03]
        lim = bins[0], bins[-1]
        lim_both = bins_both[0], bins_both[-1]

        # Alpha-alpha comparison
        if True:
            from scipy.interpolate import interp1d

            cols = {"Ding 2018 $P(k)$": c2[0], "Ding 2018 $\\xi(s)$": c2[1]}
            fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharex=True)
            labels = ["Ding 2018 $P(k)$", "Ding 2018 $\\xi(s)$"]
            k = "avg"
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
                        if j == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
                    else:
                        print(label1, label2)
                        a1 = res[label2][k].values
                        a2 = res[label1][k].values
                        c = np.sqrt(np.min(np.vstack((res[label2]["Dchi2"].values, res[label1]["Dchi2"].values)), axis=0))
                        c[np.isnan(c)] = 0
                        print(c.shape, c.min(), c.max(), c.mean())
                        m_good = c > 5
                        print("Correlation all: ", np.corrcoef(a1, a2))
                        print("Correlation 3: ", np.corrcoef(a1[m_good], a2[m_good]))

                        im = ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=2, vmax=9)

                        from mpl_toolkits.axes_grid1 import make_axes_locatable

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="3%", pad=0.0)
                        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
                        cbar = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[3, 5, 7, 9])

                        l2 = (lim[1] - lim[0]) * 1.03 + lim[0]
                        ax.set_xlim(lim[0], l2)
                        ax.set_ylim(*lim)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(0.9982, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(0.9982, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                            ax.set_yticks(ticks)
                        if i == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)
