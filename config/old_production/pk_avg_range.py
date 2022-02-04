import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter


# Checks to see if the better fits from the power spectrum are due to differences in the effective fitting range.
# Spoiler: No.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, min_k=0.03, max_k=0.21)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p, min_k=0.03, max_k=0.21)

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        model.set_default("sigma_nl", sigma_nl)
        model.set_fix_params(["om", "sigma_nl"])

        fitter.add_model_and_dataset(PowerBeutler2017(recon=r), d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(PowerSeo2016(recon=r), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(PowerDing2018(recon=r), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])
        fitter.add_model_and_dataset(PowerNoda2019(recon=r, postprocess=p), de, name=f"Noda 2019 {t}", linestyle=ls, color=cs[3])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.analysis.get_latex_table(filename=pfn + "_params.txt", parameters=[r"$\alpha$"])
        c.plotter.plot_summary(
            filename=[pfn + "_summary2.png", pfn + "_summary2.pdf"],
            extra_parameter_spacing=1.5,
            parameters=1,
            errorbar=True,
            truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982},
        )
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982}, parameters=2)
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"], truth={"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982})
