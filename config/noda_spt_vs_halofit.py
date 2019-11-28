import sys

sys.path.append("..")
from barry.config import setup
from barry.models import PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor
from barry.cosmology.camb_generator import getCambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]

    postprocess = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        rt = "Recon" if r else "Prerecon"
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=postprocess)
        n = PowerNoda2019(postprocess=postprocess, recon=r, nonlinear_type="spt")
        n2 = PowerNoda2019(postprocess=postprocess, recon=r, nonlinear_type="halofit")

        fitter.add_model_and_dataset(n, data, name=f"N19 {rt} SPT", color="r", shade_alpha=0.7, linestyle="-" if r else "--", zorder=10 if r else 2)
        fitter.add_model_and_dataset(n2, data, name=f"N19 {rt} Halofit", color="lb", shade_alpha=0.7, linestyle="-" if r else "--", zorder=10 if r else 2)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(20)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=["$\\alpha$", "$A$", "$b$"])
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.analysis.get_latex_table(filename=pfn + "_params.txt")

    # FINDINGS
    # Can get better pre-recon by swapping to Halofit NL model.
