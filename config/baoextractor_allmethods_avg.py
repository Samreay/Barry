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

    fitter = Fitter(dir_name)
    postprocess = BAOExtractor(r_s)
    for r in [True, False]:
        fix = ["om", "f"]
        t = "Recon" if r else "Prerecon"
        models = [
            PowerNoda2019(name=f"Noda {t}", postprocess=postprocess, recon=r, fix_params=fix),
            PowerSeo2016(name=f"Seo {t}", postprocess=postprocess, recon=r, fix_params=fix),
            PowerDing2018(name=f"Ding {t}", postprocess=postprocess, recon=r, fix_params=fix),
            PowerBeutler2017(name=f"Beutler {t}", postprocess=postprocess, recon=r, fix_params=fix)
        ]
        data = MockPowerSpectrum(name="BAOE mean", recon=r, min_k=0.03, max_k=0.30, postprocess=postprocess)
        for m in models:
            fitter.add_model_and_dataset(m, data)

    sampler = EnsembleSampler(temp_dir=dir_name)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data in fitter.load():
            name = f"{model.get_name()} {data.get_name()}"
            linestyle = "--" if "Prerecon" in name else "-"
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)
        c.configure(shade=True, bins=20)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



