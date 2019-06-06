import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019, PowerSeo2016, PowerBeutler2017, PowerDing2018
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
    r = True
    fix = ["om"] #, "f", "a1", "a2", "a3", "a4", "a5"]
    m = PowerBeutler2017(postprocess=postprocess, recon=r, fix_params=fix)
    d = MockPowerSpectrum(name="BAOE mean", recon=r, min_k=0.03, max_k=0.3, postprocess=postprocess)

    # sampler = EnsembleSampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)
    # fitter.set_models(m)
    # fitter.set_data(d)
    # fitter.set_sampler(sampler)
    # fitter.set_num_walkers(10)
    # fitter.fit(file, viewer=False)

    if fitter.is_laptop():

        m.set_data(d.get_data())
        p, v = m.optimize()
        print(p)
        m.plot(p)

        # from chainconsumer import ChainConsumer
        #
        # c = ChainConsumer()
        # for posterior, weight, chain, model, data in fitter.load():
        #     name = f"{model.get_name()} {data.get_name()}"
        #     linestyle = "--" if "FitOm" in name else "-"
        #     c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)
        # c.configure(shade=True, bins=20)
        # c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        # with open(pfn + "_params.txt", "w") as f:
        #     f.write(c.analysis.get_latex_table(transpose=True))



