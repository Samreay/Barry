import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019, PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.framework.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.framework.postprocessing import PureBAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    fitter = Fitter(dir_name)
    p = PureBAOExtractor(r_s)
    cs = ["#262232", "#116A71", "#48AB75", "#b7c742"]

    for r in [True]:
        t = "Recon" if r else "Prerecon"
        datae = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.03, max_k=0.30, postprocess=p)
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.03, max_k=0.30)
        for d, ls, n in zip([data, datae], ["-", ":"], ["", " Extracted"]):
            if n:
                fix = ["om", "f", "a1", "a2", "a3", "a4", "a5"]
            else:
                fix = ["om", "f"]
            fitter.add_model_and_dataset(PowerBeutler2017(postprocess=p, recon=r, fix_params=fix), d, name=f"Beutler 2017{n} {t}", linestyle=ls, color=cs[0])
            fitter.add_model_and_dataset(PowerSeo2016(postprocess=p, recon=r, fix_params=fix), d, name=f"Seo 2016{n} {t}", linestyle=ls, color=cs[1])
            fitter.add_model_and_dataset(PowerDing2018(postprocess=p, recon=r, fix_params=fix), d, name=f"Ding 2018{n} {t}", linestyle=ls, color=cs[2])
            fitter.add_model_and_dataset(PowerNoda2019(postprocess=p, recon=r, fix_params=fix), d, name=f"Noda 2019{n} {t}", linestyle=ls, color=cs[3])

    sampler = EnsembleSampler(temp_dir=dir_name)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.31, '$\\alpha$': 1.0})

    # FINDINGS: All non-noda methods similarly fit to a low alpha, whilst noda fits high. Noda would fit better if we could assume a gamma
    # value of around 4 to 5, but this makes me feel... uncomfortable.

