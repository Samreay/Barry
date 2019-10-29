import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup, weighted_avg_and_std
from barry.models import PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.fitter import Fitter
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)

        beutler_not_fixed = PowerBeutler2017(recon=r)
        beutler = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        beutler.set_default("sigma_nl", sigma_nl)
        beutler.set_fix_params(["om", "sigma_nl"])

        for i in range(999):
            d.set_realisation(i)

            fitter.add_model_and_dataset(beutler_not_fixed, d, name=f"Beutler 2017 {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(500)

    if not fitter.should_plot():
        fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        res = []
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            n = extra["name"].split(",")[0]

            alpha_mean, alpha_std = weighted_avg_and_std(chain[:, 0], weight)
            sigman_nl_mean, sigma_nl_std = weighted_avg_and_std(chain[:, 2], weight)

            res.append([n, extra["realisation"], alpha_mean, alpha_std, sigman_nl_mean])
        df = pd.DataFrame(res, columns=["name", "realisation", "alpha_mean", "alpha_std", "sigma_nl_mean"])

        import matplotlib.pyplot as plt

        plt.scatter(df["alpha_std"], df["sigma_nl_mean"], c=df["alpha_mean"])
        plt.colorbar()
        plt.xlabel("alpha std")
        plt.ylabel("sigmanl mean")
        plt.show()
