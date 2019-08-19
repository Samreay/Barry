import sys

sys.path.append("..")
from barry.framework.cosmology.camb_generator import getCambGenerator
from barry.setup import setup
from barry.framework.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, CorrBeutler2017, CorrSeo2016, CorrDing2018
from barry.framework.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC, PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, save_dims=2, remove_output=False)

    c = getCambGenerator()
    r_s, _ = c.get_data()

    sampler = EnsembleSampler(temp_dir=dir_name, num_walkers=100, num_steps=500, num_burn=300)

    for r in [True]:
        t = "Recon" if r else "Prerecon"

        d_pk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        d_xi = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=r, realisation=0)

        beutler_pk = PowerBeutler2017(recon=r)
        # seo_pk = PowerSeo2016(recon=r)
        # ding_pk = PowerDing2018(recon=r)

        beutler_xi = CorrBeutler2017()
        # seo_xi = CorrSeo2016(recon=r)
        # ding_xi = CorrDing2018(recon=r)

        for i in range(999):
            d_pk.set_realisation(i)
            d_xi.set_realisation(i)

            fitter.add_model_and_dataset(beutler_pk, d_pk, name=f"Beutler 2017 $P(k)$, mock number {i}", linestyle="-", color="p", realisation=i)
            # fitter.add_model_and_dataset(seo_pk, d_pk, name=f"Seo 2016 P(k), mock number {i}", linestyle="-", color="r")
            # fitter.add_model_and_dataset(ding_pk, d_pk, name=f"Ding 2018 P(k), mock number {i}", linestyle="-", color="lb")

            fitter.add_model_and_dataset(beutler_xi, d_xi, name=f"Beutler 2017 $\\xi(s)$, mock number {i}", linestyle=":", color="p", realisation=i)
            # fitter.add_model_and_dataset(seo_xi, d_xi, name=f"Seo 2016 corr, mock number {i}", linestyle=":", color="r")
            # fitter.add_model_and_dataset(ding_xi, d_xi, name=f"Ding 2018 corr, mock number {i}", linestyle=":", color="lb")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(400)
    if not fitter.should_plot():
        fitter.fit(file)

    if fitter.should_plot():
        import matplotlib.pyplot as plt

        import logging
        logging.info("Creating plots")

        res = {}
        for posterior, weight, chain, model, data, extra in fitter.load():
            n = extra["name"].split(",")[0]
            realisation = extra["realisation"]
            if res.get(n) is None:
                res[n] = []
            i = posterior.argmax()
            chi2 = - 2 * posterior[i]
            res[n].append([chain[:, 0].mean(), np.std(chain[:, 0]), chain[i, 0], posterior[i], chi2, -chi2])
        for label in res.keys():
            res[label] = np.array(res[label])
        ks = [l for l in res.keys() if "Smooth" not in l]

        # Define colour scheme
        c2 = ["#225465", "#5FA45E"] # ["#581d7f", "#e05286"]
        c3 = ["#2C455A", "#258E71", "#C1C64D"] # ["#501b73", "#a73b8f", "#ee8695"]
        c4 = ["#262232","#116A71","#48AB75","#D1E05B"] #["#461765", "#7b2a95", "#d54d88", "#f19a9b"]
        c5 = ["#262232", "#1F4D5C", "#0E7A6E", "#5BA561", "#C1C64D"] # ["#3c1357", "#61208d", "#a73b8f", "#e8638b", "#f4aea3"]

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        bins_both = np.linspace(0.92, 1.08, 31)
        bins = np.linspace(0.94, 1.06, 31)
        ticks = [0.97, 1.0, 1.03]
        lim = bins[0], bins[-1]
        lim_both = bins_both[0], bins_both[-1]

        # Alpha-alpha comparison
        if True:
            from scipy.interpolate import interp1d
            cols = {"Beutler 2017 $P(k)$": c2[0], "Beutler 2017 $\\xi(s)$": c2[1]}
            fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
            labels = ["Beutler 2017 $P(k)$", "Beutler 2017 $\\xi(s)$"]
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    ax = axes[i, j]
                    if i < j:
                        ax.axis('off')
                        continue
                    elif i == j:
                        h, _, _ = ax.hist(res[label1][:, 0], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[label1])
                        ax.hist(res[label1][:, 0], bins=bins, histtype="step", linewidth=1.5, color=cols[label1])
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', left=False)
                        ax.set_xlim(*lim)
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([1.0])[0]
                        ax.plot([1.0, 1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        if j == 0:
                            ax.spines['left'].set_visible(False)
                        if j == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
                    else:
                        print(label1, label2)
                        a1 = np.array(res[label2][:, 0])
                        a2 = np.array(res[label1][:, 0])
                        print("Correlation is: ", np.corrcoef(a1, a2))
                        c = np.abs(a1 - a2)
                        ax.scatter(a1, a2, s=1, c=c, cmap="viridis_r", vmin=-0.005, vmax=0.04)
                        ax.set_xlim(*lim)
                        ax.set_ylim(*lim)
                        ax.plot([0.8, 1.2], [0.8, 1.2], c="k", lw=1, alpha=0.8, ls=":")
                        ax.axvline(1.0, color="k", lw=1, ls="--", alpha=0.4)
                        ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4)
                        
                        if j != 0:
                            ax.set_yticklabels([])
                            ax.tick_params(axis='y', left=False)
                        else:
                            ax.set_ylabel(label1, fontsize=12)
                            ax.set_yticks(ticks)
                        if i == 1:
                            ax.set_xlabel(label2, fontsize=12)
                            ax.set_xticks(ticks)
            plt.subplots_adjust(hspace=0.0, wspace=0)
            fig.savefig(pfn + "_alphacomp.png", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_alphacomp.pdf", bbox_inches="tight", dpi=300, transparent=True)
