import sys

import numpy as np

sys.path.append("..")
from barry.config import setup
from barry.utils import weighted_avg_and_std
from barry.models import PowerNoda2019
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor
from barry.cosmology.camb_generator import CambGenerator
from barry.samplers import DynestySampler
from barry.fitter import Fitter


# Investigate the impact of shifting the second k anchor point
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    c = CambGenerator()
    r_s = c.get_data()["r_s"]
    fitter = Fitter(dir_name)

    ps = [
        BAOExtractor(r_s, extra_ks=(0.095, 0.13)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.14)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.15)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.16)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.17)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.18)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.19)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.20)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.21)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.22)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.23)),
        BAOExtractor(r_s, extra_ks=(0.095, 0.24)),
    ]

    recon = True
    for p in ps:
        n = f"$k = {p.extra_ks[1]:0.2f}\, h / {{\\rm Mpc}}$"
        model = PowerNoda2019(postprocess=p, recon=recon)
        data = PowerSpectrum_SDSS_DR12_Z061_NGC(min_k=0.02, max_k=0.30, postprocess=p, recon=recon)
        fitter.add_model_and_dataset(model, data, name=n)

    sampler = DynestySampler(temp_dir=dir_name)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(30)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        alphas = []
        aas = []
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():
            print(extra["name"])
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
            m, s = weighted_avg_and_std(evidence, weights=weight)
            m2, s2 = weighted_avg_and_std(chain[:, -1], weights=weight)
            alphas.append(m)
            aas.append(m2)
        print(np.std(alphas), np.std(aas))
        c.configure(shade=True, bins=25, legend_artists=True, cmap="plasma", sigmas=[0, 1, 2])
        params = ["$\\alpha$", "$A$", "$b$"]
        truth = {"$\\Omega_m$": 0.31, "$\\alpha$": 0.9982}
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth, parameters=params)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=params, figsize="COLUMN")

    # FINDINGS
    # Well, looks like where you transition from alternating indices to only using the extractor
    # has a strong impact on not only where you fit alpha, b, gamma and A, but also on their uncertainties.
    # It also impracts the degeneracy direction between alpha and A.
