import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor, PureBAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p)

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        model.set_default("sigma_nl", sigma_nl)
        model.set_fix_params(["om", "sigma_nl"])

        fitter.add_model_and_dataset(PowerBeutler2017(recon=r), d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(PowerSeo2016(recon=r), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(PowerDing2018(recon=r), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])
        fitter.add_model_and_dataset(PowerNoda2019(recon=r, postprocess=p), de, name=f"Noda 2019 {t}", linestyle=ls, color=cs[3])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(30)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")
        res = fitter.load()

        if True:
            # This is a dirty way of faking another column in the dataset so I can keep plots visually consistent.
            # Essentially resampling our weighted chain to be uniform weights.
            # Note this requires that pk_individual.py has been run first! #######################################
            ind_path = "plots/pk_individual/pk_individual_alphameans.csv"
            n = 1000000
            stds_dict = None
            if os.path.exists(ind_path):
                df = pd.read_csv(ind_path)
                cols = [c for c in df.columns if "std" in c]
                stds = df[cols]
                stds_dict = {}
                for c in cols:
                    x = np.sort(df[c])
                    print(c, np.mean(x), np.std(x))
                    cdfs = np.linspace(0, 1, x.size)
                    d = np.atleast_2d(interp1d(cdfs, x)(np.random.rand(n))).T
                    print(c, d.shape)
                    stds_dict[c.split("_pk")[0]] = d

            from chainconsumer import ChainConsumer
            import matplotlib.pyplot as plt

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            c = ChainConsumer()
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():
                # Resample to uniform weights, eugh
                m = chain.shape[0]
                weight = weight / weight.max()
                samples = None
                while samples is None or samples.shape[0] < n:
                    if samples is None:
                        samples = chain[np.random.rand(m) < weight, :]
                    else:
                        samples = np.concatenate((samples, chain[np.random.rand(m) < weight, :]))
                samples = samples[:n, :]
                if stds_dict is not None:
                    samples = np.hstack((samples, stds_dict[extra["name"]]))

                # c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
                c.add_chain(samples, parameters=model.get_labels() + [r"$\sigma_\alpha$"], **extra)
            c.configure(shade=True, bins=30, legend_artists=True)
            c.configure_truth(zorder=-10)
            c.analysis.get_latex_table(filename=pfn + "_params.txt", parameters=[r"$\alpha$"])
            c.analysis.get_latex_table(filename=pfn + "_params_all.txt", transpose=True)
            c.plotter.plot_summary(filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
            extents = {r"$\alpha$": [0.98, 1.03], r"$\sigma_\alpha$": [0.008, 0.027]}
            fig = c.plotter.plot_summary(
                filename=[pfn + "_summary2.png", pfn + "_summary2.pdf"],
                extra_parameter_spacing=1.5,
                parameters=[r"$\alpha$", r"$\sigma_\alpha$"],
                errorbar=True,
                truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982},
                extents=extents,
            )
            # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
            # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})

        # Plots the average recon mock measurements and the best-fit models from each of the models tested.
        # We'll also plot the ratio for everything against the smooth Beutler2017 model.
        if True:

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            fig, axes = plt.subplots(figsize=(5, 8), nrows=2, sharex=True, gridspec_kw={"hspace": 0.06, "height_ratios": [4.3, 1]})
            inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=axes[0], hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_ylabel(r"$P(k) / P_{\mathrm{smooth}}(k)$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for posterior, weight, chain, evidence, model, data, extra in res:
                if not "Recon" in extra["name"]:
                    continue
                if extra["name"] == "Noda 2019 Recon":
                    model.postprocess = PureBAOExtractor(r_s)
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if counter == 0:
                    ks = data[0]["ks"]
                    pk = data[0]["pk"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    rk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True, postprocess=PureBAOExtractor(r_s)).get_data()[0]
                if extra["name"] == "Noda 2019 Recon":
                    axes[1].errorbar(ks, rk["pk"], yerr=np.sqrt(np.diag(rk["cov"])), fmt="o", c="#666666", elinewidth=0.75, ms=3, zorder=0)
                    axes[1].plot(ks, bestfit, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                    axes[1].set_ylabel(r"$R(k)$", labelpad=-5)
                    axes[1].legend(frameon=False, markerfirst=False)
                else:
                    ax = fig.add_subplot(inner[counter])
                    ax.errorbar(ks, pk / smooth, yerr=err / smooth, fmt="o", c="#666666", elinewidth=0.75, ms=3, zorder=0)
                    ax.plot(ks, bestfit / smooth, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                    ax.set_ylim(0.88, 1.12)
                    ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    ax.legend(frameon=False, markerfirst=False)
                counter += 1
            axes[1].set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
            plt.savefig(pfn + "_bestfits.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_bestfits.png", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(figsize=(5, 8), nrows=2, sharex=True, gridspec_kw={"hspace": 0.04, "height_ratios": [1.5, 1]})

            for posterior, weight, chain, evidence, model, data, extra in res:
                if not "Recon" in extra["name"]:
                    continue
                if extra["name"] == "Noda 2019 Recon":
                    model.postprocess = None
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if extra["name"] == "Beutler 2017 Recon":
                    ks = data[0]["ks"]
                    pk = data[0]["pk"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    axes[0].errorbar(ks, ks * pk, yerr=ks * err, fmt="o", c="#666666", elinewidth=0.75, ms=3, zorder=0)
                    axes[1].fill_between(ks, 1.0 + err / pk, 1.0 - err / pk, color="k", zorder=0, alpha=0.15)
                axes[0].plot(ks, ks * bestfit, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
                axes[1].plot(ks, bestfit / pk, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
            axes[0].set_ylim(750.0, 1650.0)
            axes[1].set_ylim(0.95, 1.05)
            axes[0].set_ylabel(r"$kP(k) (h^{-2}\,\mathrm{Mpc^{2}})$")
            axes[1].set_ylabel(r"$\mathrm{Bestfit} / \mathrm{Data}$")
            axes[0].tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
            axes[0].legend()
            axes[1].set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
            plt.savefig(pfn + "_bestfits_2.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_bestfits_2.png", bbox_inches="tight", dpi=300, transparent=True)

        # Plots the propagator for the Seo and Ding models compared to Beutler so we can compare.
        if True:

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            fig, axes = plt.subplots(figsize=(5, 5), nrows=1, sharex=True)
            inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=axes, hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_ylabel(r"$\mathcal{C}(k,\mu)$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            muvals = np.array([0.0, 0.5, 1.0])
            for posterior, weight, chain, evidence, model, data, extra in res:
                if not "Recon" in extra["name"]:
                    continue
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                # model.mu, model.nmu = muvals, len(muvals)
                if extra["name"] == "Seo 2016 Recon":
                    damping_dd = model.get_damping_dd(p["f"], p["om"])
                    damping_ss = model.get_damping_ss(p["om"])
                    smooth_prefac = np.tile(model.camb.smoothing_kernel / p["b"], (model.nmu, 1))
                    kaiser_prefac = 1.0 + np.outer(p["f"] / p["b"] * model.mu ** 2, 1.0 - model.camb.smoothing_kernel)
                    fog = 1.0 / (1.0 + np.outer(model.mu ** 2, model.camb.ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
                    # Test for Cullan
                    d = damping_ss - damping_dd

                    propagator_Seo = (kaiser_prefac * damping_dd + smooth_prefac * (damping_ss - damping_dd)) ** 2  # * fog
                    extra_Seo = extra
                    # print(model.get_pt_data(p["om"])["sigma_dd"], model.get_pt_data(p["om"])["sigma_ss"])
                elif extra["name"] == "Ding 2018 Recon":
                    damping_dd = model.get_damping_dd(p["f"], p["om"])
                    damping_sd = model.get_damping_sd(p["f"], p["om"])
                    damping_ss = model.get_damping_ss(p["om"])
                    smooth_prefac = np.tile(model.camb.smoothing_kernel / p["b"], (model.nmu, 1))
                    bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * model.camb.ks ** 2, (model.nmu, 1))
                    kaiser_prefac = 1.0 - smooth_prefac + np.outer(p["f"] / p["b"] * model.mu ** 2, 1.0 - model.camb.smoothing_kernel) + bdelta_prefac
                    fog = 1.0 / (1.0 + np.outer(model.mu ** 2, model.camb.ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
                    propagator_Ding = (
                        (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping_dd
                        + 2.0 * kaiser_prefac * smooth_prefac * damping_sd
                        + smooth_prefac ** 2 * damping_ss
                    )  # * fog

                    extra_Ding = extra
                    # print(model.get_pt_data(p["om"])["sigma_dd_nl"], model.get_pt_data(p["om"])["sigma_sd_nl"], model.get_pt_data(p["om"])["sigma_ss_nl"])
                if extra["name"] == "Beutler 2017 Recon":
                    ks = model.camb.ks
                    fog = 1.0 / (1.0 + ks ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
                    propagator_Beutler = np.exp(-0.5 * model.camb.ks ** 2 * p["sigma_nl"] ** 2)  # * fog
                    extra_Beutler = extra
                    print(p["sigma_nl"] ** 2)
            for i, mu in enumerate(muvals):
                ax = fig.add_subplot(inner[i])
                ax.plot(
                    ks,
                    propagator_Beutler,
                    color=extra_Beutler["color"],
                    label=extra_Beutler["name"],
                    linestyle=extra_Beutler["linestyle"],
                    zorder=0,
                    linewidth=1.8,
                )
                ax.plot(ks, propagator_Seo[i, 0:], color=extra_Seo["color"], label=extra_Seo["name"], linestyle=extra_Seo["linestyle"], zorder=1, linewidth=1.8)
                ax.plot(
                    ks, propagator_Ding[i, 0:], color=extra_Ding["color"], label=extra_Ding["name"], linestyle=extra_Ding["linestyle"], zorder=2, linewidth=1.8
                )
                ax.axhline(1, c="k", lw=0.7, ls="--")
                mustr = str(r"$ \mu=%2.1lf $" % mu)
                ax.annotate(mustr, (0.93, 0.93), xycoords="axes fraction", horizontalalignment="right", verticalalignment="top", fontsize=12)
                # ax.set_xscale("log")
                # ax.set_yscale("log")
                ax.set_xlim(0.0, 0.32)
                if i == 2:
                    ax.set_ylim(0.0, 1.5)
                    ax.legend(frameon=False, markerfirst=False)
                    ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
                else:
                    ax.set_ylim(0.0, 1.1)
                    ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)

            plt.savefig(pfn + "_propagator.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_propagator.png", bbox_inches="tight", dpi=300, transparent=True)
