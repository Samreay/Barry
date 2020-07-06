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
from barry.samplers import DynestySampler, EnsembleSampler
from barry.fitter import Fitter
from barry.models.model import Correction

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)
    # sampler = EnsembleSampler(temp_dir=dir_name, num_steps=5000)
    fitter = Fitter(dir_name)

    cs = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    for r in [False]:
        t = "Recon" if r else "Prerecon"
        ls = "-"  # if r else "--"
        d = PowerSpectrum_Beutler2019_Z061_NGC(recon=r, isotropic=False, fit_poles=[0, 2], reduce_cov_factor=np.sqrt(2000.0))

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om"], correction=Correction.HARTLAP)
        model_marg = PowerBeutler2017(recon=r, isotropic=False, fix_params=["om"], correction=Correction.HARTLAP, marg=True)
        model_fixed = PowerBeutler2017(
            recon=r,
            isotropic=False,
            fix_params=[
                "om",
                "b",
                f"a{{0}}_1",
                f"a{{0}}_2",
                f"a{{0}}_3",
                f"a{{0}}_4",
                f"a{{0}}_5",
                f"a{{2}}_1",
                f"a{{2}}_2",
                f"a{{2}}_3",
                f"a{{2}}_4",
                f"a{{2}}_5",
            ],
            correction=Correction.HARTLAP,
        )
        model_fixed.set_default("b", 4.379)
        model_fixed.set_default(f"a{{0}}_1", -6278)
        model_fixed.set_default(f"a{{0}}_2", 4569)
        model_fixed.set_default(f"a{{0}}_3", -539.8)
        model_fixed.set_default(f"a{{0}}_4", 17.44)
        model_fixed.set_default(f"a{{0}}_5", -0.02013)
        model_fixed.set_default(f"a{{2}}_1", -6055)
        model_fixed.set_default(f"a{{2}}_2", 2977)
        model_fixed.set_default(f"a{{2}}_3", -138.4)
        model_fixed.set_default(f"a{{2}}_4", 15.64)
        model_fixed.set_default(f"a{{2}}_5", -0.01151)
        """model_fixed_poly = PowerBeutler2017(
            recon=r,
            isotropic=False,
            fix_params=[
                "om",
                f"a{{0}}_1",
                f"a{{0}}_2",
                f"a{{0}}_3",
                f"a{{0}}_4",
                f"a{{0}}_5",
                f"a{{2}}_1",
                f"a{{2}}_2",
                f"a{{2}}_3",
                f"a{{2}}_4",
                f"a{{2}}_5",
            ],
            correction=Correction.HARTLAP,
        )
        model_fixed_poly.set_default(f"a{{0}}_1", -6427)
        model_fixed_poly.set_default(f"a{{0}}_2", 4593)
        model_fixed_poly.set_default(f"a{{0}}_3", -599.1)
        model_fixed_poly.set_default(f"a{{0}}_4", 18.19)
        model_fixed_poly.set_default(f"a{{0}}_5", -0.02162)
        model_fixed_poly.set_default(f"a{{2}}_1", -5420)
        model_fixed_poly.set_default(f"a{{2}}_2", 2551)
        model_fixed_poly.set_default(f"a{{2}}_3", -282.9)
        model_fixed_poly.set_default(f"a{{2}}_4", 15.86)
        model_fixed_poly.set_default(f"a{{2}}_5", -0.01816)"""

        fitter.add_model_and_dataset(model, d, name=f"Full Fit", linestyle=ls, color=cs[0])
        fitter.add_model_and_dataset(model_marg, d, name=f"Analytic", linestyle=ls, color=cs[1])
        fitter.add_model_and_dataset(model_fixed, d, name=f"Fixed Bias+Poly", linestyle=ls, color=cs[2])
        # fitter.add_model_and_dataset(model_fixed_poly, d, name=f"Fixed Poly", linestyle=ls, color=cs[3])
        # fitter.add_model_and_dataset(model, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}", linestyle=ls, color=cs[0])
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
            print(np.amax(weight), np.amax(posterior), chain[np.argmax(weight)], chain[np.argmax(posterior)])
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        print(c.analysis.get_summary())
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4)
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(
            filename=[pfn + "_contour.png", pfn + "_contour.pdf"],
            truth=truth,
            parameters=["$\\alpha$", "$\\epsilon$", "$\\Sigma_s$", "$\\Sigma_{nl,||}$", "$\\Sigma_{nl,\\perp}$", "$\\beta$"],
        )
        c.plotter.plot(
            filename=[pfn + "_contour2.png", pfn + "_contour2.pdf"],
            truth=truth,
            parameters=[
                "$\\alpha$",
                "$\\epsilon$",
                "$\\Sigma_s$",
                "$\\Sigma_{nl,||}$",
                "$\\Sigma_{nl,\\perp}$",
                "$\\beta$",
                "$b$",
                "$a_{0,1}$",
                "$a_{0,2}$",
                "$a_{0,3}$",
                "$a_{0,4}$",
                "$a_{0,5}$",
                "$a_{2,1}$",
                "$a_{2,2}$",
                "$a_{2,3}$",
                "$a_{2,4}$",
                "$a_{2,5}$",
            ],
        )
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
