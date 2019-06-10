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

    fitter = Fitter(dir_name)

    p1 = BAOExtractor(r_s, mink=0.05, maxk=0.18)
    p2 = BAOExtractor(r_s, mink=0.03, maxk=0.18)
    p3 = BAOExtractor(r_s, mink=0.05, maxk=0.15)
    p4 = BAOExtractor(r_s, mink=0.05, maxk=0.25)

    names = ["0.05-0.18", "0.03-0.18", "0.05-0.15", "0.05-0.25"]

    for p, n in zip([p1, p2, p3, p4], names):
        model = PowerNoda2019(postprocess=p)
        data = MockPowerSpectrum(min_k=0.02, max_k=0.30, postprocess=p, apply_hartlap_correction=False)
        fitter.add_model_and_dataset(model, data, name=n)

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



