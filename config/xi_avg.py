import sys

sys.path.append("..")
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.postprocessing import BAOExtractor
from barry.setup import setup
from barry.framework.models import CorrBeutler2017, CorrDing2018, CorrSeo2016
from barry.framework.datasets import MockSDSSdr7CorrelationFunction
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()
    p = BAOExtractor(r_s)

    sampler = EnsembleSampler(temp_dir=dir_name, num_walkers=100, num_burn=300, num_steps=1000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#b7c742"]
    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = MockSDSSdr7CorrelationFunction(recon=r)
        fitter.add_model_and_dataset(CorrBeutler2017(), d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[0])
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
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            print(extra["name"], chain.shape, weight.shape, posterior.shape)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(filename=pfn + "_summary.png", extra_parameter_spacing=1.5, errorbar=True, truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        c.plotter.plot_summary(filename=pfn + "_summary2.png", extra_parameter_spacing=1.5, parameters=2, errorbar=True, truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})



