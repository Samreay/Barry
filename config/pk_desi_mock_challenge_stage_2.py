import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESIMockChallenge
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=True)

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)

    cs = ["#CAF270", "#84D57B", "#4AB482", "#219180", "#1A6E73", "#234B5B", "#232C3B"]

    # Power spectrum
    data = PowerSpectrum_DESIMockChallenge(recon=True, isotropic=False, fit_poles=[0, 2], min_k=0.02, max_k=0.45)
    model = PowerBeutler2017(recon=True, isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.NONE, marg="full")

    ls = "-"
    names = [f"Xinyi-std Pk", f"Pedro-std Pk", f"Baojiu-std Pk", f"Xinyi-Hada Pk", f"Hee-Jong-std Pk", f"Yu-Yu-std Pk", f"Javier-std Pk"]
    for i in range(7):
        data.set_realisation(i)
        fitter.add_model_and_dataset(model, data, name=names[i], color=cs[i], realisation=i, ls=ls)

    # Correlation Function
    data = CorrelationFunction_DESIMockChallenge(recon=True, isotropic=False, fit_poles=[0, 2])
    model = CorrBeutler2017(recon=True, isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.NONE, marg="full")

    ls = "--"
    names = [f"Xinyi-std Xi", f"Pedro-std Xi", f"Baojiu-std Xi", f"Xinyi-Hada Xi", f"Hee-Jong-std Xi", f"Yu-Yu-std Xi", f"Javier-std Xi"]
    for i in range(7):
        data.set_realisation(i)
        fitter.add_model_and_dataset(model, data, name=names[i], color=cs[i], realisation=i, ls=ls)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        output = []
        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            model.set_data(data[0])
            r_s = model.camb.get_data()["r_s"]
            print(extra["name"])

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            print(model, np.shape(alpha), np.shape(epsilon))
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            extra["linestyle"] = extra["ls"]
            extra.pop("ls", None)
            c.add_chain(df, weights=weight, **extra)

            max_post = posterior.argmax()
            chi2 = -2 * posterior[max_post]

            if "Pk" in extra["name"]:
                dof = data[0]["pk"].shape[0] - 1 - len(df.columns)
            else:
                dof = data[0]["xi"].shape[0] - 1 - len(df.columns)
            ps = chain[max_post, :]
            best_fit = {}
            for l, p in zip(model.get_labels(), ps):
                best_fit[l] = p

            mean, cov = weighted_avg_and_cov(df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]], weight, axis=0)

            c2 = ChainConsumer()
            c2.add_chain(df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]], weights=weight)
            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            if "Pk" in extra["name"]:
                output.append(
                    f"{data[0]['min_k']:5.2f}, {data[0]['max_k']:5.2f}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:5.3f}, {r_s:6.3f}, {chi2:5.3f}, {dof:4d}, {chi2 / dof:5.2f}"
                )
            else:
                output.append(
                    f"{data[0]['min_dist']:5.2f}, {data[0]['max_dist']:5.2f}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:5.3f}, {r_s:6.3f}, {chi2:5.3f}, {dof:4d}, {chi2 / dof:5.2f}"
                )

        with open(pfn + "_BAO_fitting.Barry", "w") as f:
            for l in output:
                f.write(l + "\n")
        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, statistics="mean")
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        c.plotter.plot_summary(
            filename=[pfn + "_summary.png", pfn + "_summary.pdf"],
            errorbar=True,
            truth=truth,
            parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\alpha$", "$\\epsilon$"],
            extents={
                "$\\alpha_\\parallel$": [0.961, 1.039],
                "$\\alpha_\\perp$": [0.976, 1.024],
                "$\\alpha$": [0.984, 1.016],
                "$\\epsilon$": [-0.017, 0.017],
            },
        )
        c.plotter.plot(
            filename=[pfn + "_contour.png", pfn + "_contour.pdf"], truth=truth, parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]
        )
        # c.plotter.plot(filename=[pfn + "_contour2.png", pfn + "_contour.pdf"], truth=truth, parameters=10)
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth=truth)
        c.analysis.get_latex_table(filename=pfn + "_params.txt")
