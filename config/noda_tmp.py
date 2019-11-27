import sys

sys.path.append("..")
from barry.config import setup
from barry.models import PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor
from barry.cosmology.camb_generator import getCambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter


# Check the impact of fixing parameters in the Noda model
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

        fitter.add_model_and_dataset(PowerNoda2019(postprocess=postprocess, recon=r), data)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        names2 = []
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            name = extra["name"]
            if "fixed $f$" in name:
                names2.append(name)
            i = posterior.argmax()
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        extents = None
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.plotter.plot(
            filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"],
            parameters=3,
            chains=names2,
            truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982},
            figsize="COLUMN",
            extents=extents,
        )
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 0.9982})
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        with open(pfn + "_corr.txt", "w") as f:
            f.write(c.analysis.get_correlation_table(chain="N19 Recon fixed $f$, $\\gamma_{rec}$"))
