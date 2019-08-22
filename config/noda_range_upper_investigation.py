import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.framework.postprocessing import BAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers import DynestySampler
from barry.framework.fitter import Fitter
import numpy as np


if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    fitter = Fitter(dir_name)

    ps = [
        BAOExtractor(r_s, mink=0.05, maxk=0.13),
        BAOExtractor(r_s, mink=0.05, maxk=0.15),
        BAOExtractor(r_s, mink=0.05, maxk=0.17),
        BAOExtractor(r_s, mink=0.05, maxk=0.19),
        BAOExtractor(r_s, mink=0.05, maxk=0.21),
        BAOExtractor(r_s, mink=0.05, maxk=0.23),
        BAOExtractor(r_s, mink=0.05, maxk=0.25),
        BAOExtractor(r_s, mink=0.05, maxk=0.27),
        BAOExtractor(r_s, mink=0.05, maxk=0.29),
    ]

    recon = True
    for p in ps:
        n = f"{p.mink:0.2f}-{p.maxk:0.2f}"
        print(n)
        model = PowerNoda2019(postprocess=p, recon=recon)
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(min_k=0.02, max_k=0.30, postprocess=p, recon=recon)
        fitter.add_model_and_dataset(model, data, name=n)

    sampler = DynestySampler(temp_dir=dir_name)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            print(extra["name"])
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True, cmap="plasma", )
        extents = None # {"$\\alpha$": (0.88, 1.18), "$A$": (0, 10), "$b$": (1.5, 1.8), r"$\gamma_{rec}$": (1, 8)}
        params = ["$\\alpha$", "$A$", "$b$"]
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0}, extents=extents, parameters=params)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0}, extents=extents, parameters=params, figsize="COLUMN")
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0},
        #                      extents=extents)

    # FINDINGS
    # Well, looks like where you transition from alternating indices to only using the extractor
    # has a strong impact on not only where you fit alpha, b, gamma and A, but also on their uncertainties.
    # It also impracts the degeneracy direction between alpha and A.

