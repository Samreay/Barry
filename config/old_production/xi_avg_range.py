import sys

sys.path.append("..")
from barry.config import setup
from barry.models import CorrBeutler2017, CorrDing2018, CorrSeo2016
from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

# Checks to see if the better fits from the power spectrum are due to differences in the effective fitting range.
# Spoiler: No.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]
    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=r, min_dist=20, max_dist=250)

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
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            print(extra["name"], chain.shape, weight.shape, posterior.shape)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.analysis.get_latex_table(filename=pfn + "_params.txt", parameters=[r"$\alpha$"])
        c.plotter.plot_summary(filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
        c.plotter.plot_summary(
            filename=[pfn + "_summary2.png", pfn + "_summary2.pdf"],
            extra_parameter_spacing=1.5,
            parameters=1,
            errorbar=True,
            truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982},
        )
        # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
