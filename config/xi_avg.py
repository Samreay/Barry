import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append("..")
from barry.config import setup
from barry.models import CorrBeutler2017, CorrDing2018, CorrSeo2016
from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)
    fitter = Fitter(dir_name, remove_output=False)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]
    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=r)

        # Fix sigma_nl for one of the Beutler models
        model = CorrBeutler2017()
        sigma_nl = 6.0 if r else 9.3
        model.set_default("sigma_nl", sigma_nl)
        model.set_fix_params(["om", "sigma_nl"])

        fitter.add_model_and_dataset(CorrBeutler2017(), d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(CorrSeo2016(recon=r), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(CorrDing2018(recon=r), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(30)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")
        res = fitter.load()
        if True:
            from chainconsumer import ChainConsumer

            if True:
                # This is a dirty way of faking another column in the dataset so I can keep plots visually consistent.
                # Essentially resampling our weighted chain to be uniform weights.
                # Note this requires that xi_individual.py has been run first! #######################################
                ind_path = "plots/xi_individual/xi_individual_alphameans.csv"
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
                        stds_dict[c.split("_xi")[0]] = d

                from chainconsumer import ChainConsumer

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
                c.plotter.plot_summary(
                    filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982}
                )
                extents = {r"$\alpha$": [0.975, 1.032], r"$\sigma_\alpha$": [0.01, 0.029]}
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

            fig, axes = plt.subplots(figsize=(5, 7), nrows=1, sharex=True)
            inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=axes, hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$")
            ax.set_ylabel(r"$\xi(s) - \xi_{\mathrm{smooth}}(s)\,(\times10^{-3})$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for posterior, weight, chain, evidence, model, data, extra in res:
                if not "Recon" in extra["name"]:
                    continue
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if extra["name"] == "Beutler 2017 Recon":
                    ss = data[0]["dist"]
                    xi = data[0]["xi0"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                ax = fig.add_subplot(inner[counter])
                ax.errorbar(ss, 1.0e3 * (xi - smooth), yerr=1.0e3 * err, fmt="o", c="#666666", elinewidth=0.75, ms=3, zorder=0)
                ax.plot(ss, 1.0e3 * (bestfit - smooth), color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                ax.set_ylim(-4.5, 4.5)
                ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax.legend(frameon=False, markerfirst=False, loc=1)
                counter += 1
                ax.set_ylim(-5, 6.5)
            ax.tick_params(axis="x", which="both", labelcolor="k", bottom=True, labelbottom=True)

            plt.savefig(pfn + "_bestfits.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_bestfits.png", bbox_inches="tight", dpi=300, transparent=True)

            fig, axes = plt.subplots(figsize=(5, 7), nrows=2, sharex=True, gridspec_kw={"hspace": 0.04, "height_ratios": [1.5, 1]})

            for posterior, weight, chain, evidence, model, data, extra in res:
                if not "Recon" in extra["name"]:
                    continue
                model.set_data(data)
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if extra["name"] == "Beutler 2017 Recon":
                    ss = data[0]["dist"]
                    xi = data[0]["xi0"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    axes[0].errorbar(ss, ss * ss * xi, yerr=ss * ss * err, fmt="o", c="#666666", elinewidth=0.75, ms=3, zorder=0)
                    axes[1].fill_between(ss, -1.0e3 * err, 1.0e3 * err, color="k", zorder=0, alpha=0.15)
                axes[0].plot(ss, ss * ss * bestfit, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
                axes[1].plot(ss, 1.0e3 * (bestfit - xi), color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.5)
            axes[0].set_ylim(-40.0, 85.0)
            axes[1].set_ylim(-1.6, 1.6)
            axes[0].set_ylabel(r"$s^{2}\xi(s) (h^{-2}\,\mathrm{Mpc^{2}})$")
            axes[1].set_ylabel(r"$\mathrm{Bestfit} - \mathrm{Data}\,(\times 10^{-3})$")
            axes[0].tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
            axes[0].legend(frameon=False, markerfirst=False, loc=1)
            axes[1].set_xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$")
            plt.savefig(pfn + "_bestfits_2.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_bestfits_2.png", bbox_inches="tight", dpi=300, transparent=True)
