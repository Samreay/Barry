import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib as plt
from matplotlib import cm

# Config file to fit the abacus cutsky mock means and individual realisations using Dynesty.

if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 500 live points.
    fitter = Fitter(dir_name, remove_output=True)
    sampler = DynestySampler(temp_dir=dir_name, nlive=500)

    # Set up a colour map to give a colour to each chain
    cmap = plt.cm.get_cmap("viridis")

    mocktypes = ["abacus_cutsky"]
    nzbins = [3]

    # Loop over the mocktypes
    allnames = []
    for i, (mocktype, redshift_bins) in enumerate(zip(mocktypes, nzbins)):

        # Loop over the available redshift bins for each mock type
        for z in range(redshift_bins):

            # Create the data. We'll fit mono-, quad- and hexadecapole between k=0.02 and 0.3.
            # First load up mock mean and add it to the fitting list.
            dataset = PowerSpectrum_DESI_KP4(
                recon=None,
                fit_poles=[0, 2, 4],
                min_k=0.02,
                max_k=0.30,
                mocktype=mocktype,
                redshift_bin=z + 1,
                realisation=None,
            )

            # Set up the model we'll use. Fix Omega_m and beta. 5 polynomials (default)
            # for each of the fitted multipoles. Use full analytic marginalisation for speed
            # Apply the Hartlap correction to the covariance matrix.
            model = PowerBeutler2017(
                recon=dataset.recon,
                isotropic=dataset.isotropic,
                marg="full",
                fix_params=["om", "beta"],
                poly_poles=dataset.fit_poles,
                correction=Correction.HARTLAP,
                n_poly=5,
            )

            # Create a unique name for the fit and add it to the list
            name = dataset.name + "mock mean"
            fitter.add_model_and_dataset(model, dataset, name=name)
            allnames.append(name)

            # Now add the inidividual realisations to the list
            for i in range(dataset.num_mocks):
                dataset.set_realisation(i)
                name = dataset.name + f"realisation {i}"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

    # Submit all the jobs to NERSC. We have quite a few (78), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below here is for plotting the chains once they have been run. The should_plot()
    # function will check for the presence of chains and plot if it finds them.
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        output = {}
        print(allnames)
        for name in allnames:
            fitname = " ".join(name.split()[:7])
            output[fitname] = []

        c = ChainConsumer()
        counter = 0
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            kmax = extra["name"].split(" ")[-1][9:-1]
            fitname = " ".join(extra["name"].split()[:7])

            color = plt.colors.rgb2hex(cmap(float(counter) / (len(kmaxs) - 1)))

            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            c.add_chain(df, weights=weight, color=color, posterior=posterior, **extra)

            max_post = posterior.argmax()
            chi2 = -2 * posterior[max_post]

            params = model.get_param_dict(chain[max_post])
            for name, val in params.items():
                model.set_default(name, val)

            figname = pfn + "_" + fitname + "_bestfit.pdf" if counter == 3 else None
            new_chi_squared, dof, bband, mods, smooths = model.plot(params, display=False, figname=figname)

            ps = chain[max_post, :]
            best_fit = {}
            for l, p in zip(model.get_labels(), ps):
                best_fit[l] = p

            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha_\\parallel$",
                        "$\\alpha_\\perp$",
                        "$\\Sigma_s$",
                        "$\\Sigma_{nl,||}$",
                        "$\\Sigma_{nl,\\perp}$",
                    ]
                ],
                weight,
                axis=0,
            )

            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            print(fitname)
            if "3-Poly" in fitname:
                if "No-Hexa" in fitname:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f}"
                    )
                else:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:8.1f}"
                    )
            else:
                if "No-Hexa" in fitname:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:7.3f}, {bband[5]:7.3f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:7.3f}, {bband[10]:7.3f}"
                    )
                else:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[3]:7.3f}, {mean[4]:7.3f}, {mean[2]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:7.3f}, {bband[5]:7.3f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:7.3f}, {bband[10]:7.3f}, {bband[11]:8.1f}, {bband[12]:8.1f}, {bband[13]:8.1f}, {bband[14]:7.3f}, {bband[15]:7.3f}"
                    )

            counter += 1
            if counter >= 4:
                counter = 0
                c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, legend_location=(0, -1), plot_contour=True)
                c.plotter.plot(
                    filename=[pfn + "_" + fitname + "_contour.pdf"],
                    truth=truth,
                    parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                )
                c = ChainConsumer()

        for name in output.keys():

            with open(dir_name + "/Queensland_bestfit_" + name.replace(" ", "_") + ".txt", "w") as f:
                if "3-Poly" in name:
                    if "No-Hexa" in name:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b, a0_1, a0_2, a0_3, a2_1, a2_2, a2_3\n"
                        )
                    else:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b, a0_1, a0_2, a0_3, a2_1, a2_2, a2_3, a4_1, a4_2, a4_3\n"
                        )
                else:
                    if "No-Hexa" in name:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b, a0_1, a0_2, a0_3, a0_4, a0_5, a2_1, a2_2, a2_3, a2_4, a2_5\n"
                        )
                    else:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, b, a0_1, a0_2, a0_3, a0_4, a0_5, a2_1, a2_2, a2_3, a2_4, a2_5, a4_1, a4_2, a4_3, a4_4, a4_5\n"
                        )
                for l in output[name]:
                    f.write(l + "\n")
