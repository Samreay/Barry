import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np


sys.path.append("..")
from barry.datasets.dataset_power_spectrum import PowerSpectrum_Beutler2019_Z061_NGC
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.samplers import DynestySampler, Local_Emcee
from barry.fitter import Fitter
from barry.models.model import Correction

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    # sampler = EnsembleSampler(temp_dir=dir_name, num_steps=5000)
    # sampler = Local_Emcee(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [False]:
        t = "Recon" if r else "Prerecon"
        ls = "-"  # if r else "--"
        d = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 2], reduce_cov_factor=np.sqrt(2000.0))

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om", "f"], correction=Correction.HARTLAP)
        model_marg = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om", "f"], correction=Correction.HARTLAP, marg=True)

        fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(model_marg, d, name=f"Beutler 2017 {t}", linestyle=ls, color=cs[2])
        # fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
        # fitter.add_model_and_dataset(PowerSeo2016(recon=r, isotropic=False), d, name=f"Seo 2016 {t}", linestyle=ls, color=cs[1])
        # fitter.add_model_and_dataset(PowerDing2018(recon=r, isotropic=False), d, name=f"Ding 2018 {t}", linestyle=ls, color=cs[2])

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=3)
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour.pdf"], truth=truth)
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
