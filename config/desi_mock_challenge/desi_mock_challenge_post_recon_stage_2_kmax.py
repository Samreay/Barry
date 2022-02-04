import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, PowerBeutler2017_3poly
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge_Post
from barry.fitter import Fitter
import numpy as np
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov, break_vector_and_get_blocks
import matplotlib as plt
from matplotlib import cm

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    print(pfn)
    fitter = Fitter(dir_name, remove_output=True)

    sampler = DynestySampler(temp_dir=dir_name, nlive=1000)

    names = [
        "PostRecon Iso Fix",
        "PostRecon Iso NonFix",
        "PostRecon Ani Fix",
        "PostRecon Ani NonFix",
    ]
    cmap = plt.cm.get_cmap("viridis")

    types = ["cov-fix", "cov-std", "cov-fix", "cov-std"]
    recons = ["iso", "iso", "ani", "ani"]
    kmaxs = [0.15, 0.20, 0.25, 0.30]

    counter = 0
    for h, fittype in enumerate(["Hexa", "No-Hexa"]):
        fit_poles = [0, 2] if fittype == "No-Hexa" else [0, 2, 4]
        poly_poles = [0, 2] if fittype == "No-Hexa" else [0, 2, 4]
        for m, modeltype in enumerate(["5-Poly", "3-Poly"]):
            for i, type in enumerate(types):
                for j, kmax in enumerate(kmaxs):
                    data = PowerSpectrum_DESIMockChallenge_Post(
                        isotropic=False,
                        recon=recons[i],
                        realisation=0,
                        fit_poles=fit_poles,
                        min_k=0.0075,
                        max_k=kmax,
                        num_mocks=1000,
                        covtype=type,
                        smoothtype="15",
                    )
                    model = (
                        PowerBeutler2017(
                            recon=data.recon,
                            isotropic=data.isotropic,
                            marg="full",
                            fix_params=["om"],
                            poly_poles=poly_poles,
                            correction=Correction.NONE,
                        )
                        if modeltype == "5-Poly"
                        else PowerBeutler2017_3poly(
                            recon=data.recon,
                            isotropic=data.isotropic,
                            marg="full",
                            fix_params=["om"],
                            poly_poles=poly_poles,
                            correction=Correction.NONE,
                        )
                    )
                    name = names[i] + " " + fittype + " " + modeltype + str(r" $k_{max}=%3.2lf$" % kmax)
                    print(name)
                    fitter.add_model_and_dataset(model, data, name=name)
                    counter += 1

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        output = {}
        namelist = []
        for name in names:
            for h, fittype in enumerate(["Hexa", "No-Hexa"]):
                for m, modeltype in enumerate(["5-Poly", "3-Poly"]):
                    output[name + " " + fittype + " " + modeltype] = []
                    namelist.append(name + " " + fittype + " " + modeltype)

        c = ChainConsumer()
        counter = np.zeros(len(4 * names))
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            kmax = extra["name"].split(" ")[-1][9:-1]
            fitname = " ".join(extra["name"].split()[:5])
            print(kmax, fitname)
            nameindex = [i for i, n in enumerate(namelist) if n == fitname][0]

            color = plt.colors.rgb2hex(cmap(float(counter[nameindex]) / (len(kmaxs) - 1)))

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

            # Ensures we return the window convolved model
            icov_m_w = model.data[0]["icov_m_w"]
            model.data[0]["icov_m_w"][0] = None

            ks = model.data[0]["ks"]
            err = np.sqrt(np.diag(model.data[0]["cov"]))
            mod, mod_odd, polymod, polymod_odd, _ = model.get_model(params, model.data[0], data_name=data[0]["name"])

            if model.marg:
                mask = data[0]["m_w_mask"]
                mod_fit, mod_fit_odd = mod[mask], mod_odd[mask]

                len_poly = len(model.data[0]["ks"]) if model.isotropic else len(model.data[0]["ks"]) * len(model.data[0]["fit_poles"])
                polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
                for nn in range(np.shape(polymod)[0]):
                    polymod_fit[nn], polymod_fit_odd[nn] = polymod[nn, mask], polymod_odd[nn, mask]

                bband = model.get_ML_nuisance(
                    model.data[0]["pk"],
                    mod_fit,
                    mod_fit_odd,
                    polymod_fit,
                    polymod_fit_odd,
                    model.data[0]["icov"],
                    model.data[0]["icov_m_w"],
                )
                mod += mod_odd + bband @ (polymod + polymod_odd)
                mod_fit += mod_fit_odd + bband @ (polymod_fit + polymod_fit_odd)

                # print(len(model.get_active_params()) + len(bband))
                # print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
                new_chi_squared = -2.0 * model.get_chi2_likelihood(
                    model.data[0]["pk"],
                    mod_fit,
                    np.zeros(mod_fit.shape),
                    model.data[0]["icov"],
                    model.data[0]["icov_m_w"],
                    num_mocks=model.data[0]["num_mocks"],
                    num_params=len(model.get_active_params()) + len(bband),
                )
                alphas = model.get_alphas(params["alpha"], params["epsilon"])
                print(new_chi_squared, len(model.data[0]["pk"]) - len(model.get_active_params()) - len(bband), bband)

            model.data[0]["icov_m_w"] = icov_m_w
            dof = data[0]["pk"].shape[0] - 1 - len(df.columns)

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
                        "$\\beta$",
                        "$\\Sigma_{nl,||}$",
                        "$\\Sigma_{nl,\\perp}$",
                    ]
                ],
                weight,
                axis=0,
            )

            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            print(corr, print(c.analysis.get_correlations()))
            if "3-Poly" in fitname:
                if "No-Hexa" in fitname:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[4]:7.3f}, {mean[5]:7.3f}, {mean[2]:7.3f}, {mean[3]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f}"
                    )
                else:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[4]:7.3f}, {mean[5]:7.3f}, {mean[2]:7.3f}, {mean[3]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:8.1f}"
                    )
            else:
                if "No-Hexa" in fitname:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[4]:7.3f}, {mean[5]:7.3f}, {mean[2]:7.3f}, {mean[3]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:7.3f}, {bband[5]:7.3f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:7.3f}, {bband[10]:7.3f}"
                    )
                else:
                    output[fitname].append(
                        f"{kmax:3s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[4]:7.3f}, {mean[5]:7.3f}, {mean[2]:7.3f}, {mean[3]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:7.3f}, {bband[5]:7.3f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:7.3f}, {bband[10]:7.3f}, {bband[11]:8.1f}, {bband[12]:8.1f}, {bband[13]:8.1f}, {bband[14]:7.3f}, {bband[15]:7.3f}"
                    )

            counter[nameindex] += 1

        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, legend_location=(0, -1), plot_contour=True)
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}

        # c.analysis.get_latex_table(filename=pfn + "_params.txt")
        for name in namelist:

            if "PostRecon Iso NonFix" not in name:
                continue

            chainnames = [name + str(r" $k_{max}=%3.2lf$" % kmax) for kmax in kmaxs]

            """c.plotter.plot_summary(
                filename=[pfn + "_" + name + "_summary.png"],
                errorbar=True,
                truth=truth,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                extents={
                    "$\\alpha_\\parallel$": [0.987, 1.012],
                    "$\\alpha_\\perp$": [0.987, 1.007],
                },
                chains=chainnames,
            )"""
            c.plotter.plot(
                filename=[pfn + "_" + name + "_contour.pdf"],
                truth=truth,
                parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
                chains=chainnames,
            )
            # c.plotter.plot(filename=[pfn + "_" + name + "_contour2.pdf"], truth=truth, chains=chainnames)

            with open(dir_name + "/Queensland_bestfit_" + name.replace(" ", "_") + ".txt", "w") as f:
                if "3-Poly" in name:
                    if "No-Hexa" in name:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_1, a0_2, a0_3, a2_1, a2_2, a2_3\n"
                        )
                    else:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_1, a0_2, a0_3, a2_1, a2_2, a2_3, a4_1, a4_2, a4_3\n"
                        )
                else:
                    if "No-Hexa" in name:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_1, a0_2, a0_3, a0_4, a0_5, a2_1, a2_2, a2_3, a2_4, a2_5\n"
                        )
                    else:
                        f.write(
                            "# kmax, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_1, a0_2, a0_3, a0_4, a0_5, a2_1, a2_2, a2_3, a2_4, a2_5, a4_1, a4_2, a4_3, a4_4, a4_5\n"
                        )
                for l in output[name]:
                    f.write(l + "\n")
