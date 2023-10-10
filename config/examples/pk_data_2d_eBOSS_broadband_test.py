import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import NautilusSampler
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_eBOSS_LRGpCMASS
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

    datasets = [
        PowerSpectrum_eBOSS_LRGpCMASS(
            realisation="data", galactic_cap="ngc", recon="iso", isotropic=False, fit_poles=[0, 2], min_k=0.02, max_k=0.30, num_mocks=999
        ),
        PowerSpectrum_eBOSS_LRGpCMASS(
            realisation="data", galactic_cap="sgc", recon="iso", isotropic=False, fit_poles=[0, 2], min_k=0.02, max_k=0.30, num_mocks=999
        ),
    ]

    # Standard Beutler Model
    model_poly = PowerBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        marg="full",
        broadband_type="poly",
        n_poly=[-1, 0, 1, 2, 3, 4],
    )
    model_poly.set_default("sigma_nl_par", 7.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_poly.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_poly.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    model_spline = PowerBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        marg="full",
        delta=2.0,
    )
    model_spline.set_default("sigma_nl_par", 7.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_spline.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_spline.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    for d in datasets:
        fitter.add_model_and_dataset(model_poly, d, name=d.name + " poly")
        fitter.add_model_and_dataset(model_spline, d, name=d.name + " spline")

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

        extents = {
            "NGC": [
                [0.95, 1.05],
                [-0.10, 0.05],
            ],
            "SGC": [
                [0.80, 1.20],
                [-0.2, 0.20],
            ],
        }
        for skybin in range(2):

            sky = "NGC" if skybin == 0 else "SGC"

            # Read in the original eBOSS chain from Hector
            infile = f"../../barry/data/sdss_dr16_lrgpcmass_pk/mcmcBAOANISO_output_P02_kmax030_BAOANISO_postrecon_DATAv7_cov7_{sky}.txt"
            og_chain = pd.read_csv(infile, header=None, delim_whitespace=True).to_numpy().T
            alp, eps = og_chain[2] ** (1.0 / 3.0) * og_chain[3] ** (2.0 / 3.0), (og_chain[2] / og_chain[3]) ** (1.0 / 3.0) - 1.0
            c[skybin].add_chain(
                np.c_[alp, eps, og_chain[2], og_chain[3]],
                parameters=["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                weights=og_chain[0],
                name="Gil-Marin et. al., 2020",
                color="k",
                shade_alpha=0.5,
            )

            c[skybin].configure(
                shade=True,
                legend_artists=True,
                # legend_location=(0, -1),
                legend_kwargs={"fontsize": 10},
                plot_contour=True,
                zorder=[4, 3, 5],
            )
            axes = (
                c[skybin]
                .plotter.plot(
                    # filename=pfn + f"{sky}_contour.pdf",
                    # parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                    parameters=["$\\alpha$", "$\\epsilon$"],
                    extents=extents[sky],
                )
                .get_axes()
            )
            results = c[skybin].analysis.get_summary(parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"])
            print(c[skybin].analysis.get_latex_table(parameters=["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"]))
            print(results)
            plt.savefig(pfn + f"_{sky}_contour.pdf", bbox_inches="tight")
