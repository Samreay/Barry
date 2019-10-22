import sys

sys.path.append("..")
from barry.cosmology.camb_generator import CambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import CorrBeutler2017, CorrDing2018, CorrSeo2016
from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

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

        if False:
            from chainconsumer import ChainConsumer

            c = ChainConsumer()
            for posterior, weight, chain, model, data, extra in fitter.load():
                c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
                print(extra["name"], chain.shape, weight.shape, posterior.shape)
            c.configure(shade=True, bins=30, legend_artists=True)
            c.analysis.get_latex_table(filename=pfn + "_params.txt", parameters=[r"$\alpha$"])
            c.plotter.plot_summary(filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 1.0})
            c.plotter.plot_summary(
                filename=[pfn + "_summary2.png", pfn + "_summary2.pdf"],
                extra_parameter_spacing=1.5,
                parameters=1,
                errorbar=True,
                truth={"$\\Omega_m$": 0.31, "$\\alpha$": 1.0},
            )
            # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
            # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})

        # Plots the average recon mock measurements and the best-fit models from each of the models tested.
        # We'll also plot the ratio for everything against the smooth Beutler2017 model.
        if True:

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig, axes = plt.subplots(figsize=(6, 8), nrows=1, sharex=True, gridspec_kw={"hspace": 0.06, "height_ratios": [4.3, 1]})
            inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=axes[0], hspace=0.04)
            ax = plt.subplot(inner[0:])
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_ylabel(r"$\xi(s) - \xi_{\mathrm{smooth}}(s)$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

            counter = 0
            for posterior, weight, chain, model, data, extra in fitter.load():
                if not "Recon" in extra["name"]:
                    continue
                model.set_data(data)
                print(data[0].keys())
                p = model.get_param_dict(chain[np.argmax(posterior)])
                smooth = model.get_model(p, data[0], smooth=True)
                bestfit = model.get_model(p, data[0])
                if counter == 0:
                    ss = data[0]["dist"]
                    xi = data[0]["xi0"]
                    err = np.sqrt(np.diag(data[0]["cov"]))
                    ax = fig.add_subplot(inner[counter])
                    ax.errorbar(ss, xi - smooth, yerr=err / smooth, fmt="o", c="k", ms=4, zorder=0)
                    ax.plot(ss, bestfit - smooth, color=extra["color"], label=extra["name"], linestyle=extra["linestyle"], zorder=1, linewidth=1.8)
                    ax.set_ylim(0.88, 1.12)
                    ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    ax.legend()
                counter += 1
            axes[1].set_xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$")
            plt.savefig(pfn + "_bestfits.pdf")
