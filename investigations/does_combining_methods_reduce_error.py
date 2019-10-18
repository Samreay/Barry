import sys

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerSeo2016, PowerBeutler2017, PowerDing2018, PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.fitter import Fitter
import numpy as np
import pandas as pd

if __name__ == "__main__":
    pfn, dir_name, file = setup("../config/pk_individual.py")
    fitter = Fitter(dir_name, save_dims=2, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()[0]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p, realisation=0)

        beutler_not_fixed = PowerBeutler2017(recon=r)
        beutler = PowerBeutler2017(recon=r)
        sigma_nl = 6.0 if r else 9.3
        beutler.set_default("sigma_nl", sigma_nl)
        beutler.set_fix_params(["om", "sigma_nl"])

        seo = PowerSeo2016(recon=r)
        ding = PowerDing2018(recon=r)
        noda = PowerNoda2019(recon=r, postprocess=p)

        for i in range(999):
            d.set_realisation(i)
            de.set_realisation(i)

            fitter.add_model_and_dataset(beutler_not_fixed, d, name=f"Beutler 2017 {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler, d, name=f"Beutler 2017 Fixed $\\Sigma_{{nl}}$ {t}, mock number {i}", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(seo, d, name=f"Seo 2016 {t}, mock number {i}", linestyle=ls, color="r", realisation=i)
            fitter.add_model_and_dataset(ding, d, name=f"Ding 2018 {t}, mock number {i}", linestyle=ls, color="lb", realisation=i)
            fitter.add_model_and_dataset(noda, de, name=f"Noda 2019 {t}, mock number {i}", linestyle=ls, color="o", realisation=i)

    import logging

    logging.info("Computing covariance matrix")

    res = {}
    for posterior, weight, chain, model, data, extra in fitter.load():
        n = extra["name"].split(",")[0]
        if res.get(n) is None:
            res[n] = []
        i = posterior.argmax()
        chi2 = -2 * posterior[i]
        res[n].append([np.average(chain[:, 0], weights=weight), np.std(chain[:, 0]), chain[i, 0], posterior[i], chi2, -chi2, extra["realisation"]])
    for label in res.keys():
        res[label] = pd.DataFrame(res[label], columns=["avg", "std", "max", "posterior", "chi2", "Dchi2", "realisation"])

    ks = list(res.keys())
    all_ids = pd.concat(tuple([res[l][["realisation"]] for l in ks]))
    counts = all_ids.groupby("realisation").size().reset_index()
    max_count = counts.values[:, 1].max()
    good_ids = counts.loc[counts.values[:, 1] == max_count, ["realisation"]]

    for label, df in res.items():
        res[label] = pd.merge(good_ids, df, how="left", on="realisation")

    labels = ["Beutler 2017 Recon", "Beutler 2017 Fixed $\\Sigma_{nl}$ Recon", "Seo 2016 Recon", "Ding 2018 Recon", "Noda 2019 Recon"]
    res2d = np.empty((len(labels), len(res[labels[0]]["avg"])))
    for i, label in enumerate(labels):
        res2d[i, 0:] = res[label]["avg"]
    mean = np.mean(res2d, axis=1)
    cov = np.cov(res2d)
    corr = np.corrcoef(res2d)

    print(np.sqrt(np.diag(cov)), corr)

    # Compute the consensus value using the equation of Winkler1981, Sanchez2016
    from scipy import linalg

    cov_inv = linalg.inv(cov)
    # print(mean.shape, cov_inv.shape)
    sigma_c = np.sum(cov_inv)
    combined = np.sum(cov_inv * mean) / sigma_c
    print(mean, combined)
    print(np.sqrt(np.diag(cov)), 1.0 / np.sqrt(sigma_c))
    print(1.0 / np.sqrt(sigma_c * np.diag(cov)))

    # Answer: Yes, by between 5-10%
