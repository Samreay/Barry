import logging
import sys
sys.path.append("../..")
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.models.baoextractor_power_poly import BAOExtractor
from barry.framework.datasets.mock_bao_extractor import MockBAOExtractorPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.config.base import setup
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    models = [BAOExtractor(r_s, name="BAO Extractor")]

    datas = [MockBAOExtractorPowerSpectrum(r_s, name="BAOE mean", recon=True, min_k=0.02, max_k=0.30)]

    sampler = EnsembleSampler(num_steps=1500, num_burn=500, temp_dir=dir_name, save_interval=30)

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_data(*datas)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(50)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data in fitter.load():
            name = f"{model.get_name()} {data.get_name()}"
            linestyle = "--" if "FitOm" in name else "-"
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)
        c.configure(shade=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



