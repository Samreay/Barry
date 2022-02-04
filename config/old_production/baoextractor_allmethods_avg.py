import sys

sys.path.append("..")
from barry.config import setup
from barry.models import PowerNoda2019, PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor
from barry.cosmology.camb_generator import getCambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter

# We want to verify the behaviour of other models when the BAO extractor technique is applied to them
# We also want to see what the behaviour is if we keep the polynomial terms enabled or disable them.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]

    fitter = Fitter(dir_name)
    p = BAOExtractor(r_s)
    cs = ["#262232", "#116A71", "#48AB75", "#b7c742"]

    for r in [True]:
        t = "Recon" if r else "Prerecon"
        datae = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.03, max_k=0.30, postprocess=p)
        for ls, n in zip(["-", ":"], ["", " (No Poly)"]):
            if n:
                fix = ["om", "f", "a1", "a2", "a3", "a4", "a5"]
            else:
                fix = ["om", "f"]
            fitter.add_model_and_dataset(PowerBeutler2017(postprocess=p, recon=r, fix_params=fix), datae, name=f"Beutler 2017{n}", linestyle=ls, color=cs[0])
            fitter.add_model_and_dataset(PowerSeo2016(postprocess=p, recon=r, fix_params=fix), datae, name=f"Seo 2016{n}", linestyle=ls, color=cs[1])
            fitter.add_model_and_dataset(PowerDing2018(postprocess=p, recon=r, fix_params=fix), datae, name=f"Ding 2018{n}", linestyle=ls, color=cs[2])
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=p, recon=r), datae, name=f"Noda 2019", color=cs[3])

    sampler = DynestySampler(temp_dir=dir_name, nlive=300)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
