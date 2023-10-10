import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import NautilusSampler
from barry.config import setup
from barry.models import CorrBeutler2017
from barry.datasets.dataset_correlation_function import CorrelationFunction_eBOSS_LRGpCMASS
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
import matplotlib.pyplot as plt

# Run an optimisation on each of the post-recon SDSS DR12 mocks. Then compare to the pre-recon mocks
# to compute the cross-correlation between BAO parameters and pre-recon measurements

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name)

    sampler = NautilusSampler(temp_dir=dir_name)

    dataset = CorrelationFunction_eBOSS_LRGpCMASS(
        realisation="data", recon="iso", isotropic=False, fit_poles=[0, 2], min_dist=50.0, max_dist=150.0
    )

    # Standard Beutler Model
    model_poly = CorrBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.NONE,
        marg="full",
        n_poly=[-2, -1, 0],
    )
    model_poly.set_default("sigma_nl_par", 7.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_poly.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_poly.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    model_spline = CorrBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.NONE,
        marg="full",
        n_poly=[0, 2, 4],
    )
    model_spline.set_default("sigma_nl_par", 7.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_spline.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_spline.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    fitter.add_model_and_dataset(model_poly, dataset, name=dataset.name + " poly")
    fitter.add_model_and_dataset(model_spline, dataset, name=dataset.name + " spline")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        counter = 0
        c = [ChainConsumer(), ChainConsumer()]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            fitname = "_".join([extra["name"].split()[4], extra["name"].split()[6]])
            skybin = 0 if "ngc" in fitname.lower() else 1

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            extra.pop("name", None)
            print(" ".join(fitname.split("_")))
            c[skybin].add_chain(df, weights=weight, posterior=posterior, name=" ".join(fitname.split("_")), **extra)

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior.argmax()
            params = df.loc[max_post]
            params_dict = model.get_param_dict(chain[max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)

            new_chi_squared, dof, bband, mods, smooths = model.plot(
                params_dict, figname=pfn + "_" + fitname + "_bestfit.pdf", display=False
            )

        alp_per, alp_par = 17.85823691865007 / 17.436, 19.32575373059217 / 20.194
        alp_cov = np.array(
            [
                [0.2838176386340292 / 20.194**2, -0.05831820341302727 / (17.436 * 20.194)],
                [-0.0583182034130273 / (17.436 * 20.194), 0.1076634008565565 / 17.436**2],
            ]
        )
        alp, eps = alp_per ** (2.0 / 3.0) * alp_par ** (1.0 / 3.0), (alp_par / alp_per) ** (1.0 / 3.0) - 1.0
        jac = np.array(
            [
                [
                    (1.0 / 3.0) * alp_per ** (2.0 / 3.0) * alp_par ** (-2.0 / 3.0),
                    (2.0 / 3.0) * alp_per ** (-1.0 / 3.0) * alp_par ** (1.0 / 3.0),
                ],
                [
                    (1.0 / 3.0) * alp_par ** (-2.0 / 3.0) * alp_per ** (-1.0 / 3.0),
                    (-1.0 / 3.0) * alp_par ** (1.0 / 3.0) * alp_per ** (-4.0 / 3.0),
                ],
            ]
        )
        alp_cov2 = jac @ alp_cov @ jac.T
        print(alp, eps, alp_par, alp_per)
        print(np.sqrt(np.diag(alp_cov2)), np.sqrt(np.diag(alp_cov)))

        aperp, apar = [1.042, 0.992], [0.947, 0.996]
        aperp_err, apar_err = [0.024, 0.038], [0.026, 0.113]
        for skybin in range(1):

            """c[skybin].add_covariance(
                [alp_par, alp_per],
                alp_cov,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                name="Bautista et. al., 2020",
                color="k",
                shade_alpha=0.5,
            )"""
            c[skybin].add_covariance(
                [alp, eps], alp_cov2, parameters=["$\\alpha$", "$\\epsilon$"], name="Bautista et. al., 2021", color="k", shade_alpha=0.5
            )

            sky = "NGC" if skybin == 0 else "SGC"
            c[skybin].configure(
                shade=True,
                bins=20,
                legend_artists=True,
                max_ticks=4,
                # legend_location=(0, -1),
                legend_kwargs={"fontsize": 10},
                plot_contour=True,
                zorder=[3, 5, 4],
            )
            axes = (
                c[skybin]
                .plotter.plot(
                    # filename=pfn + f"{sky}_contour.pdf",
                    # parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                    parameters=["$\\alpha$", "$\\epsilon$"],
                )
                .get_axes()
            )
            # axes[0].axvspan(apar[skybin] - apar_err[skybin], apar[skybin] + apar_err[skybin], color="k", alpha=0.1, zorder=1)
            # axes[2].axhspan(aperp[skybin] - aperp_err[skybin], aperp[skybin] + aperp_err[skybin], color="k", alpha=0.1, zorder=1)
            # axes[2].axvspan(apar[skybin] - apar_err[skybin], apar[skybin] + apar_err[skybin], color="k", alpha=0.1, zorder=1)
            # axes[3].axhspan(aperp[skybin] - aperp_err[skybin], aperp[skybin] + aperp_err[skybin], color="k", alpha=0.1, zorder=1)
            results = c[skybin].analysis.get_summary(parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"])
            print(c[skybin].analysis.get_latex_table(parameters=["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"]))
            print(results)
            # plt.tight_layout()
            plt.savefig(pfn + f"_{sky}_contour.pdf", bbox_inches="tight")
