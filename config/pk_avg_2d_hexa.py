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
    # dir_name = dir_name + "nlive_1500/"

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    # sampler = EnsembleSampler(temp_dir=dir_name, num_steps=5000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-"  # if r else "--"
        d_quad = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 2], reduce_cov_factor=np.sqrt(2000.0))
        d_odd = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 1, 2], reduce_cov_factor=np.sqrt(2000.0))
        d_hexa = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 2, 4], reduce_cov_factor=np.sqrt(2000.0))
        d_all = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 1, 2, 3, 4], reduce_cov_factor=np.sqrt(2000.0))

        # Fix sigma_nl for one of the Beutler models
        model_quad = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.HARTLAP, marg="full"
        )
        model_odd = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 1, 2], correction=Correction.HARTLAP, marg="full"
        )
        model_hexa = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 2, 4], correction=Correction.HARTLAP, marg=True
        )
        model_all = PowerBeutler2017(
            recon=r, isotropic=False, fix_params=["om"], poly_poles=[0, 1, 2, 3, 4], correction=Correction.HARTLAP, marg=True
        )

        fitter.add_model_and_dataset(model_quad, d_quad, name=f"P_{{0}}+P_{{2}}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model_odd, d_odd, name=f"P_{{0}}+P_{{1}}+P_{{2}}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(model_hexa, d_hexa, name=f"P_{{0}}+P_{{2}}+P_{{4}}", linestyle=ls, color=cs[2])
        fitter.add_model_and_dataset(model_all, d_all, name=f"P_{{0}}+P_{{1}}+P_{{2}}+P_{{3}}+P_{{4}}", linestyle=ls, color=cs[2])

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
                c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            c.configure(shade=True, legend_artists=True, max_ticks=4, sigmas=[0, 1, 2], label_font_size=12, tick_font_size=12, kde=True)
            truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0.0}
            c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True)
            c.plotter.plot(
                figsize="COLUMN",
                filename=[pfn + "_contour.png", pfn + "_contour.pdf"],
                parameters=["$\\alpha$", "$\\epsilon$"],
                extents={"$\\alpha$": (0.980, 1.015), "$\\epsilon$": (-0.02, 0.035)},
                truth=truth,
                chains=[f"P_{{0}}+P_{{2}}", f"Mono+Quad+Hexa With Poly", f"Mono+Quad+Hexa Without Poly"],
            )

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
