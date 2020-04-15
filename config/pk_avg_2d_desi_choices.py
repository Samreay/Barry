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
from barry.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method

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

        # Changing fitting range and sigma values to match Hee-Jong
        d = PowerSpectrum_DESIMockChallenge0_Z01(recon=r, isotropic=False, realisation="data", min_k=0.001, max_k=0.30)
        model = PowerBeutler2017(recon=r, isotropic=False, correction=Correction.NONE)
        model.set_default("sigma_nl_par", 6.2)
        model.set_default("sigma_nl_perp", 2.9)
        model.set_default("sigma_s", 0.0)
        model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"])

        fitter.add_model_and_dataset(model, d, name=f"Hee-Jong $\\Sigma_{{nl}}$")

        # Now change linear and smooth spectra to Hee-Jong's inputs
        pklin = np.array(pd.read_csv("../barry/data/desi_mock_challenge_0/mylinearmatterpkL900.dat", delim_whitespace=True, header=None))
        pksmooth = np.array(
            pd.read_csv("../barry/data/desi_mock_challenge_0/Psh_mylinearmatterpkL900.dat", delim_whitespace=True, header=None, skiprows=2)
        )
        model2 = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
        model2.set_default("sigma_nl_par", 6.2)
        model2.set_default("sigma_nl_perp", 2.9)
        model2.set_default("sigma_s", 0.0)
        model2.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"])
        model2.set_data(d.get_data())
        model2.kvals = pklin[:, 0]
        model2.pksmooth = smooth(model2.kvals, pklin[:, 1])
        model2.pkratio = pklin[:, 1] / model2.pksmooth - 1.0

        fitter.add_model_and_dataset(model2, d, name=f"Hee-Jong Pk")

        # Now change linear and smooth spectra to Hee-Jong's inputs
        model3 = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
        model3.set_default("sigma_nl_par", 6.2)
        model3.set_default("sigma_nl_perp", 2.9)
        model3.set_default("sigma_s", 0.0)
        model3.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"])
        model3.set_data(d.get_data())
        model3.kvals = pklin[:, 0]
        model3.pkratio = pklin[:, 1] / pksmooth[:, 1] - 1.0
        model3.pksmooth = pksmooth[:, 1]

        fitter.add_model_and_dataset(model3, d, name=f"Hee-Jong Pk+Pk$_{{sm}}$")

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
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, legend_kwargs={"fontsize": 8})
        # truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha_{par}$": 1.102, "$\\alpha_{perp}$": 1.034}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(
            filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=["$\\alpha_{par}$", "$\\alpha_{perp}$"]
        )
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour.pdf"], truth=truth, parameters=10)
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
