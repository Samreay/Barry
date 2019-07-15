import sys
sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019
from barry.framework.datasets import MockSDSSPowerSpectrum
from barry.framework.postprocessing import BAOExtractor
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter
import numpy as np


if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s)

    sampler = EnsembleSampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        rt = "Recon" if r else "Prerecon"
        data = MockSDSSPowerSpectrum(recon=r, postprocess=postprocess, reduce_cov_factor=np.sqrt(1000))
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f", "gamma", "b"]), data, name=f"Node {rt} fixed om, f, gamma, b", linestyle="-" if r else "--", color="o")
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f", "gamma"]), data, name=f"Node {rt} fixed om, f, gamma", linestyle="-" if r else "--", color="r")
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om", "f"],), data, name=f"Node {rt} fixed om, f", linestyle="-" if r else "--", color="lb")
        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r, fix_params=["om"],), data, name=f"Node {rt} fixed om", linestyle="-" if r else "--", color="p")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file, viewer=False)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table())

    # FINDINGS
    # So turns out that fixing all these parameters really helps get good constraints.
    # Both the choice of b and gamma entirely determine where alpha will fit.
    # Fixing gamma, f and b gives constraints 4 times better than letting them free
    # Really fixing b is what is driving down uncertainty.
