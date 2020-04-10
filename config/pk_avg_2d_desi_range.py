import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np


sys.path.append("..")
from barry.utils import weighted_avg_and_std
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge0_Z01
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerBeutler2017
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

        # Fix sigma_nl for one of the Beutler models
        model = PowerBeutler2017(recon=r, isotropic=False, correction=Correction.NONE)
        model.set_default("sigma_nl_par", 10.9)
        model.set_default("sigma_nl_perp", 5.98)
        model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp"])

        for mink, maxk in [(0.02, 0.3), (0.02, 0.4), (0.02, 0.5), (0.03, 0.3), (0.03, 0.5), (0.03, 0.4), (0.04, 0.3), (0.04, 0.4), (0.04, 0.5)]:
            d = PowerSpectrum_DESIMockChallenge0_Z01(recon=r, isotropic=False, realisation="data", min_k=mink, max_k=maxk)
            fitter.add_model_and_dataset(model, d, name=f"B17 Fixed {t}, k = ({mink:0.2f}, {maxk:0.2f})")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        output = []
        bestfit_model = []
        pks = []
        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            model.set_data(data)
            p = model.get_param_dict(chain[np.argmax(posterior)])
            print(p, model.get_alphas(p["alpha"], p["epsilon"]))
            bestfit = model.get_model(p, data[0])
            pk_smooth_lin, pk_ratio = model.compute_basic_power_spectrum(p["om"])
            pks.append((pk_smooth_lin * (1.0 + pk_ratio), pk_smooth_lin))
            bestfit_model.append((data[0]["ks"], bestfit))
            strout = str(pfn + "_bestfit_%3.2lf_%3.2lf.pdf" % (data[0]["min_k"], data[0]["max_k"]))
            model.plot(p, figname=strout)

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            c.add_chain(df, weights=weight, **extra)

            max_post = posterior.argmax()
            chi2 = -2 * posterior[max_post]

            dof = data[0]["pk"].shape[0] - 1 - len(df.columns)
            ps = chain[max_post, :]
            best_fit = {}
            for l, p in zip(model.get_labels(), ps):
                best_fit[l] = p

            mean_par, std_par = weighted_avg_and_std(alpha_par, weight)
            mean_per, std_per = weighted_avg_and_std(alpha_perp, weight)

            c2 = ChainConsumer()
            c2.add_chain(df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]], weights=weight)
            _, corr = c2.analysis.get_correlations()
            corr = corr[1, 0]
            output.append(
                f"{data[0]['min_k']:5.2f}, {data[0]['max_k']:5.2f}, {mean_par:5.3f}, {mean_per:5.3f}, {std_par:5.3f}, {std_per:5.3f}, {corr:5.3f}, {r_s:6.3f}, {chi2:5.3f}, {dof:4d}, {chi2/dof:5.2f}"
            )

        np.savetxt(pfn + "_Pk_linear_CAMB.dat", np.c_[model.camb.ks, pks[0][0]], fmt="%g        %g", header="k     pk")
        np.savetxt(pfn + "_Pk_linear_CAMB_smooth.dat", np.c_[model.camb.ks, pks[0][1]], fmt="%g     %g", header="k     pk_smooth")
        np.savetxt(
            pfn + "_bestfit_model.dat",
            np.c_[bestfit_model[0][0], bestfit_model[0][1][: len(bestfit_model[0][0])], bestfit_model[0][1][len(bestfit_model[0][0]) :]],
            fmt="%g  %g  %g",
            header="k     pk0     pk2",
        )

        """with open(pfn + "_BAO_fitting_DC.v0.Barry", "w") as f:
            for l in output:
                f.write(l + "\n")
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, statistics="mean")
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0}
        c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], errorbar=True, truth=truth)
        c.plotter.plot(filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=3)
        c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour.pdf"], truth=truth, parameters=10)
        c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")"""
