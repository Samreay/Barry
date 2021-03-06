import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np


sys.path.append("..")
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge0_Z01
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.samplers import DynestySampler
from barry.fitter import Fitter
from barry.models.model import Correction

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [False]:
        t = "Recon" if r else "Prerecon"
        d = PowerSpectrum_DESIMockChallenge0_Z01(recon=r, isotropic=False, realisation="data")

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r, isotropic=False, correction=Correction.NONE)
        model.set_default("sigma_nl_par", 10.9)
        model.set_default("sigma_nl_perp", 5.98)
        model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp"])

        fitter.add_model_and_dataset(PowerBeutler2017(recon=r, isotropic=False, correction=Correction.NONE, fix_params=["om"]), d, name=f"Beutler 2017 {t}")
        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}")
        # fitter.add_model_and_dataset(PowerSeo2016(recon=r, isotropic=False), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        # fitter.add_model_and_dataset(PowerDing2018(recon=r, isotropic=False), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            chain_conv = chain
            chain_conv[:, 0], chain_conv[:, 2] = model.get_alphas(chain[:, 0], chain[:, 2])
            parameters = model.get_labels()
            parameters[0] = r"$\alpha_{par}$"
            parameters[2] = r"$\alpha_{perp}$"
            c.add_chain(chain, weights=weight, parameters=parameters, **extra)
            max_post = posterior.argmax()
            ps = chain_conv[max_post, :]
            for l, p in zip(parameters, ps):
                print(l, p)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        # truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha_{par}$": 1.0, "$\\alpha_{perp}$": 1.0}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=3)
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour.pdf"], truth=truth, parameters=10)
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
