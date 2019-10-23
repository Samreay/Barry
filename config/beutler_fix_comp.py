import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)

    c = getCambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True]:  # , False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        dmean = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=None)

        beutler = PowerBeutler2017(recon=r)

        beutler_fixed = PowerBeutler2017(recon=r)
        beutler_fixed.set_data(dmean.get_data())
        ps, minv = beutler_fixed.optimize()
        sigma_nl = ps["sigma_nl"]
        beutler_fixed.set_default("sigma_nl", sigma_nl)
        beutler_fixed.set_fix_params(["om", "sigma_nl"])

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        for i in range(999):
            d.set_realisation(i)
            fitter.add_model_and_dataset(beutler, d, name=f"B17, mock number {i}", linestyle=ls, color="p")
            fitter.add_model_and_dataset(beutler_fixed, d, name=f"B17 Fixed $\\Sigma_{{nl}}$, mock number {i}", linestyle=ls, color="p")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(300)
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
            if rese.get(n) is None:
                rese[n] = []
            res[n].append(chain[:, 0].mean())
            rese[n].append(np.std(chain[:, 0]))

        # Define colour scheme
        c2 = ["#225465", "#5FA45E"]  # ["#581d7f", "#e05286"]
        c3 = ["#2C455A", "#258E71", "#C1C64D"]  # ["#501b73", "#a73b8f", "#ee8695"]
        c4 = ["#262232", "#116A71", "#48AB75", "#D1E05B"]  # ["#461765", "#7b2a95", "#d54d88", "#f19a9b"]
        c5 = ["#262232", "#1F4D5C", "#0E7A6E", "#5BA561", "#C1C64D"]  # ["#3c1357", "#61208d", "#a73b8f", "#e8638b", "#f4aea3"]
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
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
            axes[0].axvline(1.0, color="k", lw=1, ls="--", alpha=0.6)
            axes[1].axvline(1.0, color="k", lw=1, ls="--", alpha=0.6)
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
        if True:
            from scipy.interpolate import interp1d

            bins = np.linspace(0.94, 1.06, 31)
            cols = {"B17": c4[0], "B17 Fixed $\\Sigma_{nl}$": c4[2]}
            fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharex=True)
            labels = ["B17", "B17 Fixed $\\Sigma_{nl}$"]
            # labels = ["Beutler Prerecon", "Seo Prerecon", "Ding Prerecon", "Noda Prerecon"]
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis("off")
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(res[label1], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1])
                        ax.hist(res[label1], bins=bins, histtype="step", linewidth=1.5, color=cols[label1])
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                        ax.set_xlim(0.95, 1.07)
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([1.0])[0]
                        ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        if j == 0:
                            ax.spines["left"].set_visible(False)
                        if j == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks([0.95, 1.0, 1.05])
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label1])
                        a2 = np.array(res[label2])
                        # c = blend_hex(cols[label1], cols[label2])
                        c = np.abs(a1 - a2)
                        ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.005, vmax=0.03)
                        ax.set_xlim(0.95, 1.07)
                        ax.set_ylim(0.95, 1.07)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(1.0, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4)

                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis="y", left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                            ax.set_yticks([0.95, 1.0, 1.05])
                        if i == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks([0.95, 1.0, 1.05])
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)
