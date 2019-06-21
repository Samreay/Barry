import sys

sys.path.append("..")
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.postprocessing import BAOExtractor
from barry.setup import setup
from barry.framework.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.framework.datasets import MockSDSSPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()
    p = BAOExtractor(r_s)

    sampler = EnsembleSampler(temp_dir=dir_name, num_walkers=100)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = MockSDSSPowerSpectrum(name=f"SDSS {t}", recon=r,  reduce_cov_factor=np.sqrt(1000))
        de = MockSDSSPowerSpectrum(name=f"SDSS {t}", recon=r,  reduce_cov_factor=np.sqrt(1000), postprocess=p)

        fitter.add_model_and_dataset(PowerBeutler2017(recon=r), d, name=f"Beutler {t}", linestyle=ls, color="p")
        fitter.add_model_and_dataset(PowerSeo2016(recon=r), d, name=f"Seo {t}", linestyle=ls, color="r")
        fitter.add_model_and_dataset(PowerDing2018(recon=r), d, name=f"Ding {t}", linestyle=ls, color="lb")
        fitter.add_model_and_dataset(PowerNoda2019(recon=r, postprocess=p), de, name=f"Noda {t}", linestyle=ls, color="o")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table())
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})



