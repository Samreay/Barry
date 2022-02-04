import sys


sys.path.append("..")
from barry.config import setup
from barry.models import PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor
from barry.cosmology.camb_generator import CambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter


# Investigate the impact of shifting the minimum k value at which the extractor kicks in
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s = c.get_data()["r_s"]

    fitter = Fitter(dir_name)

    ps = [BAOExtractor(r_s, mink=0.03), BAOExtractor(r_s, mink=0.04), BAOExtractor(r_s, mink=0.05), BAOExtractor(r_s, mink=0.06), BAOExtractor(r_s, mink=0.07)]

    recon = True
    for p in ps:
        n = f"$k = {p.mink:0.2f}\, h / {{\\rm Mpc}}$"
        model = PowerNoda2019(postprocess=p, recon=recon)
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(min_k=0.02, max_k=0.30, postprocess=p, recon=recon)
        fitter.add_model_and_dataset(model, data, name=n)

    sampler = DynestySampler(temp_dir=dir_name)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(30)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            print(extra["name"])
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True, cmap="plasma", sigmas=[0, 1])
        params = ["$\\alpha$", "$A$", "$b$"]
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(
            filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982}, parameters=params
        )
        c.plotter.plot(
            filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982}, parameters=params, figsize="COLUMN"
        )

    # FINDINGS
    # The minimum k value doesnt shift things much.
