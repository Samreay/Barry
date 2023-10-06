import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import NautilusSampler
from barry.config import setup
from barry.utils import weighted_avg_and_cov
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12, PowerSpectrum_eBOSS_LRGpCMASS
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
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1,
            realisation="data",
            galactic_cap="ngc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=1,
            realisation="data",
            galactic_cap="sgc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2,
            realisation="data",
            galactic_cap="ngc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=2,
            realisation="data",
            galactic_cap="sgc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=3,
            realisation="data",
            galactic_cap="ngc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
        PowerSpectrum_SDSS_DR12(
            redshift_bin=3,
            realisation="data",
            galactic_cap="sgc",
            recon="iso",
            isotropic=False,
            fit_poles=[0, 2],
            min_k=0.02,
            max_k=0.30,
            num_mocks=999,
        ),
    ]

    # Standard Beutler Model
    model_poly = PowerBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om", "sigma_nl_par", "sigma_nl_perp"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        marg="full",
        broadband_type="poly",
        n_poly=[-3, -2, -1, 0, 1],
    )
    model_poly.set_default("sigma_nl_par", 4.0)
    model_poly.set_default("sigma_nl_perp", 2.0)

    model_spline = PowerBeutler2017(
        recon="iso",
        isotropic=False,
        fix_params=["om", "sigma_nl_par", "sigma_nl_perp"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        marg="full",
    )
    model_spline.set_default("sigma_nl_par", 4.0)
    model_spline.set_default("sigma_nl_perp", 2.0)

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
        c = [ChainConsumer()] * 6
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            fitname = "_".join([extra["name"].split()[4], extra["name"].split()[6]])
            skybin = 0 if "ngc" in fitname.lower() else 1
            redbin = int(extra["name"].split()[2][-1]) - 1
            fitbin = skybin * 3 + redbin
            print(fitbin, fitname)

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            extra.pop("name", None)
            c[skybin].add_chain(df, weights=weight, posterior=posterior, name=" ".join(fitname.split()), **extra)

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

        for skybin in range(2):
            for redbin in range(3):

                sky = "NGC" if skybin == 0 else "SGC"
                red = "redshift_bin " + str(redbin + 1)
                c[skybin].configure(shade=True, bins=20, legend_artists=True, max_ticks=4, plot_contour=True, zorder=[5, 4])
                truth = {
                    "$\\Omega_m$": 0.3121,
                    "$\\alpha$": 1.0,
                    "$\\epsilon$": 0,
                    # "$\\alpha_\\perp$": aperp[skybin],
                    # "$\\alpha_\\parallel$": apar[skybin],
                }
                axes = (
                    c[skybin]
                    .plotter.plot(
                        # filename=pfn + f"{sky}_contour.pdf",
                        truth=truth,
                        parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                    )
                    .get_axes()
                )
                results = c[skybin].analysis.get_summary(parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"])
                print(results)
                plt.tight_layout()
                plt.savefig(pfn + f"_{sky}_{red}_contour.pdf")
