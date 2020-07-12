import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np


sys.path.append("..")
from barry.datasets.dataset_power_spectrum import PowerSpectrum_Beutler2019_Z061_NGC
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.samplers import DynestySampler, EnsembleSampler
from barry.fitter import Fitter
from barry.models.model import Correction

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    # sampler = EnsembleSampler(temp_dir=dir_name, num_steps=5000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [False]:
        t = "Recon" if r else "Prerecon"
        ls = "-"  # if r else "--"
        d = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 2], reduce_cov_factor=np.sqrt(2000.0))

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om"], correction=Correction.HARTLAP)
        model_marg = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om"], correction=Correction.HARTLAP, marg=True)
        model_fixed = PowerBeutler2017(
            recon=r,
            isotropic=False,
            fix_params=[
                "om",
                "b",
                f"a{{0}}_1",
                f"a{{0}}_2",
                f"a{{0}}_3",
                f"a{{0}}_4",
                f"a{{0}}_5",
                f"a{{2}}_1",
                f"a{{2}}_2",
                f"a{{2}}_3",
                f"a{{2}}_4",
                f"a{{2}}_5",
            ],
            correction=Correction.HARTLAP,
        )
        model_fixed.set_default("b", 4.379)
        model_fixed.set_default(f"a{{0}}_1", -6278)
        model_fixed.set_default(f"a{{0}}_2", 4569)
        model_fixed.set_default(f"a{{0}}_3", -539.8)
        model_fixed.set_default(f"a{{0}}_4", 17.44)
        model_fixed.set_default(f"a{{0}}_5", -0.02013)
        model_fixed.set_default(f"a{{2}}_1", -6055)
        model_fixed.set_default(f"a{{2}}_2", 2977)
        model_fixed.set_default(f"a{{2}}_3", -138.4)
        model_fixed.set_default(f"a{{2}}_4", 15.64)
        model_fixed.set_default(f"a{{2}}_5", -0.01151)
        """model_fixed_poly = PowerBeutler2017(
            recon=r,
            isotropic=False,
            fix_params=[
                "om",
                f"a{{0}}_1",
                f"a{{0}}_2",
                f"a{{0}}_3",
                f"a{{0}}_4",
                f"a{{0}}_5",
                f"a{{2}}_1",
                f"a{{2}}_2",
                f"a{{2}}_3",
                f"a{{2}}_4",
                f"a{{2}}_5",
            ],
            correction=Correction.HARTLAP,
        )
        model_fixed_poly.set_default(f"a{{0}}_1", -6427)
        model_fixed_poly.set_default(f"a{{0}}_2", 4593)
        model_fixed_poly.set_default(f"a{{0}}_3", -599.1)
        model_fixed_poly.set_default(f"a{{0}}_4", 18.19)
        model_fixed_poly.set_default(f"a{{0}}_5", -0.02162)
        model_fixed_poly.set_default(f"a{{2}}_1", -5420)
        model_fixed_poly.set_default(f"a{{2}}_2", 2551)
        model_fixed_poly.set_default(f"a{{2}}_3", -282.9)
        model_fixed_poly.set_default(f"a{{2}}_4", 15.86)
        model_fixed_poly.set_default(f"a{{2}}_5", -0.01816)"""

        fitter.add_model_and_dataset(model, d, name=f"Full Fit", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model_marg, d, name=f"Analytic", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(model_fixed, d, name=f"Fixed Bias+Poly", linestyle=ls, color=cs[2])
        # fitter.add_model_and_dataset(model_fixed_poly, d, name=f"Fixed Poly", linestyle=ls, color=cs[3])
        # fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        # fitter.add_model_and_dataset(PowerSeo2016(recon=r, isotropic=False), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        # fitter.add_model_and_dataset(PowerDing2018(recon=r, isotropic=False), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        if True:

            from chainconsumer import ChainConsumer

            c = ChainConsumer()
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                if extra["name"] == f"Fixed Bias+Poly":
                    continue
                c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            c.configure(shade=True, legend_artists=True, max_ticks=4, sigmas=[0, 1, 2, 3], label_font_size=16, tick_font_size=12, kde=True)
            truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
            c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
            c.plotter.plot(
                figsize="PAGE",
                filename=[pfn + "_contour.png", pfn + "_contour.pdf"],
                truth=truth,
                parameters=["$\\alpha$", "$\\epsilon$", "$\\Sigma_s$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\beta$"],
            )
            c.plotter.plot(
                filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"],
                truth=truth,
                parameters=[
                    "$\\alpha$",
                    "$\\epsilon$",
                    "$\\Sigma_s$",
                    "$\\Sigma_{nl,||}$",
                    "$\\Sigma_{nl,\\perp}$",
                    "$\\beta$",
                    "$b$",
                    "$a_{0,1}$",
                    "$a_{0,2}$",
                    "$a_{0,3}$",
                    "$a_{0,4}$",
                    "$a_{0,5}$",
                    "$a_{2,1}$",
                    "$a_{2,2}$",
                    "$a_{2,3}$",
                    "$a_{2,4}$",
                    "$a_{2,5}$",
                ],
            )
            c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
            c.analysis.get_latex_table(filename=pfn + "_params.txt")

        if False:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                if extra["name"] == f"Full Fit":
                    model.set_data(data)
                    p = model.get_param_dict(chain[np.argmax(posterior)])
                    for name, val in p.items():
                        model.set_default(name, val)
                    # p, minv = model.optimize(close_default=1.0e6, niter=100, maxiter=20000)
                    # print(p, minv)

            ks = data[0]["ks"]
            err = np.sqrt(np.diag(data[0]["cov"]))
            model.set_data(data)
            mod = model.get_model(p, data[0])[0]
            smooth = model.get_model(p, data[0], smooth=True)[0]
            print(mod, smooth)

            names = [f"pk{n}" for n in model.data[0]["fit_poles"]]

            # Split up the different multipoles if we have them
            if len(err) > len(ks):
                assert len(err) % len(ks) == 0, f"Cannot split your data - have {len(err)} points and {len(ks)} modes"
            errs = [col for i, col in enumerate(err.reshape((-1, len(ks)))) if i in model.data[0]["fit_poles"]]
            mods = [col for i, col in enumerate(mod.reshape((-1, len(ks)))) if i in model.data[0]["fit_poles"]]
            smooths = [col for i, col in enumerate(smooth.reshape((-1, len(ks)))) if i in model.data[0]["fit_poles"]]

            fig, axes = plt.subplots(figsize=(12, 4), ncols=2, sharey=False, gridspec_kw={"wspace": 0.20})

            mfcs = ["#666666", "w"]
            lines = ["-", "--"]

            ax1 = fig.add_subplot(axes[0])
            ax1.set_ylim(0.0, 2000.0)
            ax1.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=14)
            ax1.set_ylabel(r"$k \times P(k)\,(h^{-2}\,\mathrm{Mpc^{2}})$", fontsize=14)
            ax1.tick_params(labelsize=12)
            ax1.annotate(
                r"$\boldsymbol{\mathrm{BOSS-DR12\,Mocks}}$"
                "\n"
                r"$\boldsymbol{\mathrm{Pre-reconstruction}}$"
                "\n"
                r"$\boldsymbol{0.5 < z < 0.7}$",
                xy=(0.98, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=14,
            )

            ax2 = fig.add_subplot(axes[1])
            ax2.spines["top"].set_color("none")
            ax2.spines["bottom"].set_color("none")
            ax2.spines["left"].set_color("none")
            ax2.spines["right"].set_color("none")
            ax2.tick_params(axis="both", which="both", labelcolor="none", top=False, bottom=False, left=False, right=False)
            ax2.set_ylabel(r"$k^{2} \times [P_{\ell}(k) - P_{\ell}^{\mathrm{smooth}}(k)]\,(h^{-1}\,\mathrm{Mpc})$", fontsize=14)

            inner = gridspec.GridSpecFromSubplotSpec(len(names), 1, subplot_spec=axes[1], hspace=0.25)
            for i, (inn, err, mod, smooth, name, line, mfc) in enumerate(zip(inner, errs, mods, smooths, names, lines, mfcs)):
                ax1.errorbar(ks, ks * data[0][name], yerr=ks * err, fmt="o", ms=5, c="#666666", mfc=mfc)
                ax1.plot(ks, ks * mod, c=extra["color"], ls=line, linewidth=1.5)

                ax2 = fig.add_subplot(inn)
                ax2.errorbar(ks, ks ** 2 * (data[0][name] - smooth), yerr=ks ** 2 * err, fmt="o", ms=5, c="#666666")
                ax2.plot(ks, ks ** 2 * (mod - smooth), c=extra["color"], ls=line, linewidth=1.5)
                if i == 0:
                    # ax2.set_ylim(-100.0, 100.0)
                    ax2.annotate(r"$\boldsymbol{P_{0}(k)}$", xy=(0.90, 0.92), xycoords="axes fraction", ha="right", va="top", fontsize=14)
                elif i == 1:
                    # ax2.set_ylim(-65.0, 65.0)
                    ax2.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=14)
                    ax2.annotate(r"$\boldsymbol{P_{2}(k)}$", xy=(0.90, 0.92), xycoords="axes fraction", ha="right", va="top", fontsize=14)
                ax2.tick_params(labelsize=12)

            fig.savefig(pfn + "_bestfits.pdf", bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(pfn + "_bestfits.png", bbox_inches="tight", dpi=300, transparent=True)
