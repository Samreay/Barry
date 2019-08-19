import sys
sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.framework.postprocessing import BAOExtractor
from barry.framework.cosmology.camb_generator import getCambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np


if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s)

    sampler = EnsembleSampler(temp_dir=dir_name, num_steps=1000)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        rt = "Recon" if r else "Prerecon"
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=postprocess)
        n = PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f", "gamma", "b"])
        n.param_dict["b"].default = 2.019 if r else 2.093
        fitter.add_model_and_dataset(n, data, name=f"N19 {rt} fixed $f$, $\\gamma$, $b$", linestyle="-" if r else "--", color="o", shade_alpha=0.7)
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f", "gamma"]), data, name=f"N19 {rt} fixed $f$, $\\gamma$", linestyle="-" if r else "--", color="r", shade_alpha=0.3)
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f"],), data, name=f"N19 {rt} fixed $f$", linestyle="-" if r else "--", color="p", shade_alpha=0.1)
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om"],), data, name=f"N19 {rt}", linestyle="-" if r else "--", color="lb", shade_alpha=0.1)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        names2 = []
        for posterior, weight, chain, model, data, extra in fitter.load():
            print(model.get_names())
            print(chain.mean(axis=0))
            name = extra["name"]
            if "fixed $f$" in name:
                names2.append(name)
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True)
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"], truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0}, parameters=3, chains=names2, figsize="COLUMN")
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.analysis.get_latex_table(filename=pfn + "_params.txt")

    # FINDINGS
    # So turns out that fixing all these parameters really helps get good constraints.
    # Both the choice of b and gamma entirely determine where alpha will fit.
    # Fixing gamma, f and b gives constraints 4 times better than letting them free
    # Really fixing b is what is driving down uncertainty.
