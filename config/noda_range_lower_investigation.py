import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import MockSDSSdr12PowerSpectrum
from barry.framework.postprocessing import BAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np


if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    fitter = Fitter(dir_name)

    ps = [
        BAOExtractor(r_s, mink=0.02, maxk=0.15),
        BAOExtractor(r_s, mink=0.03, maxk=0.15),
        BAOExtractor(r_s, mink=0.04, maxk=0.15),
        BAOExtractor(r_s, mink=0.05, maxk=0.15),
        BAOExtractor(r_s, mink=0.06, maxk=0.15),
        BAOExtractor(r_s, mink=0.07, maxk=0.15),
    ]

    for p in ps:
        n = f"{p.mink:0.2f}-{p.maxk:0.2f}"
        model = PowerNoda2019(postprocess=p, recon=True)
        data = MockSDSSdr12PowerSpectrum(min_k=0.02, max_k=0.30, postprocess=p, recon=True)
        fitter.add_model_and_dataset(model, data, name=n)

    sampler = EnsembleSampler(temp_dir=dir_name)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20)
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0}, extents=extents)
        c.analysis.get_latex_table(transpose=True, filename=pfn + "_params.txt")
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})



