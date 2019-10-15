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
    r_s = c.get_data()[0]

    postprocess = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        rt = "Recon" if r else "Prerecon"
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=postprocess)
        n = PowerNoda2019(postprocess=postprocess, recon=r, nonlinear_type="spt")
        n2 = PowerNoda2019(postprocess=postprocess, recon=r, nonlinear_type="halofit")

        fitter.add_model_and_dataset(n, data, name=f"N19 {rt} SPT", color="r", shade_alpha=0.7)
        fitter.add_model_and_dataset(n2, data, name=f"N19 {rt} Halofit", color="o", shade_alpha=0.7)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(20)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        names2 = []
        for posterior, weight, chain, model, data, extra in fitter.load():
            # print(model.get_names())
            # print(chain.mean(axis=0))
            name = extra["name"]
            if "fixed $f$" in name:
                names2.append(name)
            i = posterior.argmax()
            print(name, model.get_names(), chain[i, :])
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        # extents = {"$\\alpha$": (0.963, 1.06)}
        extents = None
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0})
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0})
        c.plotter.plot(
            filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"],
            parameters=3,
            chains=names2,
            truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0},
            figsize="COLUMN",
            extents=extents,
        )
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0})
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        with open(pfn + "_corr.txt", "w") as f:
            f.write(c.analysis.get_correlation_table(chain="N19 Recon fixed $f$, $\\gamma_{rec}$"))

    # FINDINGS
    # So turns out that fixing all these parameters really helps get good constraints.
    # Both the choice of b and gamma entirely determine where alpha will fit.
    # Really fixing b is what is driving down uncertainty.
