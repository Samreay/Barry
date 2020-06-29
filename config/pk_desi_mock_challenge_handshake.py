import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np


sys.path.append("..")
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge_Handshake
from barry.cosmology.camb_generator import getCambGenerator
from barry.config import setup
from barry.models import PowerSpectrumFit, PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    d = PowerSpectrum_DESIMockChallenge_Handshake(min_k=0.005, max_k=0.3, isotropic=False, realisation="data", fit_poles=[0, 2])

    fitter.add_model_and_dataset(PowerBeutler2017(isotropic=False), d, name=f"Beutler 2017 Prerecon", color=cs[0])
    fitter.add_model_and_dataset(PowerSeo2016(isotropic=False), d, name=f"Seo 2016 Prerecon", color=cs[1])
    fitter.add_model_and_dataset(PowerDing2018(isotropic=False), d, name=f"Ding 2018 Prerecon", color=cs[2])
    fitter.add_model_and_dataset(PowerNoda2019(isotropic=False), d, name=f"Noda 2019 Prerecon", color=cs[3])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")
        res = fitter.load()

        from chainconsumer import ChainConsumer
        import copy

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in res:
            chain_conv = copy.deepcopy(chain)
            chain_conv[:, 0], chain_conv[:, 2] = model.get_alphas(chain[:, 0], chain[:, 2])
            parameters = model.get_labels()
            parameters[0] = r"$\alpha_{par}$"
            parameters[2] = r"$\alpha_{perp}$"
            c.add_chain(chain_conv, weights=weight, parameters=parameters, **extra)
            max_post = posterior.argmax()
            ps = chain_conv[max_post, :]
            if extra["name"] == "Beutler 2017 Prerecon":
                for l, p in zip(parameters, ps):
                    print(l, p)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, legend_kwargs={"fontsize": 8})
        truth = {"$\\alpha_{par}$": 1.0, "$\\alpha_{perp}$": 1.0}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(
            filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters={"$\\alpha_{par}$", "$\\alpha_{perp}$"}
        )
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"], truth=truth)
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")

        # Plots the measurements and the best-fit models from each of the models tested.
        # We'll also plot the ratio for everything against the smooth model.
        if True:

            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            fig1, axes1 = plt.subplots(figsize=(5, 8), nrows=len(res), sharex=True, gridspec_kw={"hspace": 0.08})
            fig2, axes2 = plt.subplots(figsize=(5, 8), nrows=len(res), sharex=True, gridspec_kw={"hspace": 0.08})
            labels = [
                r"$k \times P(k)\,(h^{-2}\,\mathrm{Mpc^{2}})$",
                r"$k \times (P(k) - P_{\mathrm{smooth}}(k))\,(h^{-2}\,\mathrm{Mpc^{2}})$",
            ]
            for fig, label in zip([fig1, fig2], labels):
                ax = fig.add_subplot(111, frameon=False)
                ax.set_ylabel(label)
                ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
                ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for posterior, weight, chain, evidence, model, data, extra in res:
                ks = data[0]["ks"]
                err = np.sqrt(np.diag(data[0]["cov"]))
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                mod = model.get_model(p, data[0])
                smooth = model.get_model(p, data[0], smooth=True)

                """if extra["name"] == "Beutler 2017 Prerecon":
                    print(ks, mod[: len(ks)], mod[len(ks) :])
                    np.savetxt(
                        "Barry_bestfit_model.txt",
                        np.c_[ks, mod[: len(ks)], mod[len(ks) : 2 * len(ks)], mod[2 * len(ks) :]],
                        fmt="%g  %g  %g  %g",
                        header="k      P0      P2       P4",
                    )

                from barry.utils import break_vector_and_get_blocks

                pk_model_fit = break_vector_and_get_blocks(mod, len(data[0]["poles"]), data[0]["fit_pole_indices"])
                diff = data[0]["pk"] - pk_model_fit
                chi2 = diff.T @ data[0]["icov"] @ diff
                print(chi2, (data[0]["num_mocks"] / 2) * np.log(1 + chi2 / (data[0]["num_mocks"] - 1)))
                exit()"""

                # Split up the different multipoles if we have them
                if len(err) > len(ks):
                    assert len(err) % len(ks) == 0, f"Cannot split your data - have {len(err)} points and {len(ks)} modes"
                errs = [col for col in err.reshape((-1, len(ks)))]
                mods = [col for col in mod.reshape((-1, len(ks)))]
                smooths = [col for col in smooth.reshape((-1, len(ks)))]

                names = [f"pk{n}" for n in model.data[0]["fit_poles"]]

                ax1 = fig1.add_subplot(axes1[counter])
                axes = fig2.add_subplot(axes2[counter])
                axes.spines["top"].set_color("none")
                axes.spines["bottom"].set_color("none")
                axes.spines["left"].set_color("none")
                axes.spines["right"].set_color("none")
                axes.tick_params(axis="both", which="both", labelcolor="none", top=False, bottom=False, left=False, right=False)

                mfcs = ["#666666", "w"]
                lines = ["-", "--"]
                inner = gridspec.GridSpecFromSubplotSpec(1, len(names), subplot_spec=axes2[counter], wspace=0.08)
                for i, (inn, err, mod, smooth, name, line, mfc) in enumerate(zip(inner, errs, mods, smooths, names, lines, mfcs)):

                    ax1.errorbar(ks, ks * data[0][name], yerr=ks * err, fmt="o", ms=4, c="#666666", mfc=mfc)
                    ax1.plot(ks, ks * mod, c=extra["color"], ls=line)
                    if counter != (len(res) - 1):
                        ax1.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    ax1.annotate(extra["name"], xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top")

                    ax2 = fig2.add_subplot(inn)
                    ax2.errorbar(ks, ks * (data[0][name] - smooth), yerr=ks * err, fmt="o", ms=4, c="#666666")
                    ax2.plot(ks, ks * (mod - smooth), c=extra["color"])
                    ax2.set_ylim(-80.0, 80.0)
                    if counter == 0:
                        if i == 0:
                            ax2.set_title(r"$P_{0}(k)$")
                        elif i == 1:
                            ax2.set_title(r"$P_{2}(k)$")
                    if counter != (len(res) - 1):
                        ax2.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)

                    if i != 0:
                        ax2.tick_params(axis="y", which="both", labelcolor="none", bottom=False, labelbottom=False)
                        ax2.annotate(extra["name"], xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top")

                counter += 1
            fig1.savefig(pfn + "_bestfits.pdf", bbox_inches="tight", dpi=300, transparent=True)
            fig1.savefig(pfn + "_bestfits.png", bbox_inches="tight", dpi=300, transparent=True)
            fig2.savefig(pfn + "_bestfits_2.pdf", bbox_inches="tight", dpi=300, transparent=True)
            fig2.savefig(pfn + "_bestfits_2.png", bbox_inches="tight", dpi=300, transparent=True)
