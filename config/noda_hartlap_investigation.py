import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.postprocessing import BAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        model = PowerNoda2019(postprocess=postprocess)
        data1 = MockPowerSpectrum(recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess, apply_hartlap_correction=True)
        data2 = MockPowerSpectrum(recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess, apply_hartlap_correction=False)

        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        fitter.add_model_and_dataset(model, data1, name=f"Noda {t} hartlap applied", linestyle=ls)
        fitter.add_model_and_dataset(model, data2, name=f"Noda {t} no hartlap", linestyle=ls)

    sampler = EnsembleSampler(temp_dir=dir_name)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(30)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



