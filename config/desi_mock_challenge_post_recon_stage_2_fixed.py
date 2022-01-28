import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017_3poly, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge_Post
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESIMockChallenge_Post
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

    sampler = DynestySampler(temp_dir=dir_name, nlive=500)

    names = [
        "PostRecon Yu Yu Pk Iso 15",
        "PostRecon Yu Yu Pk Iso 15 Fixed Sigma",
        "PostRecon Yu Yu Pk Ani 15",
        "PostRecon Yu Yu Pk Ani 15 Fixed Sigma",
        "PostRecon Yu Yu Xi Iso 15",
        "PostRecon Yu Yu Xi Iso 15 Fixed Sigma",
        "PostRecon Yu Yu Xi Ani 15",
        "PostRecon Yu Yu Xi Ani 15 Fixed Sigma",
    ]
    cmap = plt.cm.get_cmap("viridis")

    types = ["rec-iso15", "rec-ani15"]
    recons = [True, True]
    recon_types = ["iso", "ani"]
    realisations = [1, 1]

    # Post-Recon Yu-Yu 15 only, first power spectrum, then correlation function
    counter = 0
    for i, type in enumerate(types):
        for j in range(realisations[i]):
            print(i, type, j, realisations[i])
            data = PowerSpectrum_DESIMockChallenge_Post(
                isotropic=False, recon=recons[i], realisation=j, fit_poles=[0, 2], min_k=0.0074, max_k=0.1976, num_mocks=1000, type=type
            )
            model = PowerBeutler2017_3poly(
                recon=recon_types[i], isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.NONE, marg="full"
            )
            fitter.add_model_and_dataset(model, data, name=names[counter])
            counter += 1

            model = PowerBeutler2017_3poly(
                recon=recon_types[i],
                isotropic=False,
                fix_params=["om", "sigma_nl_par", "sigma_nl_perp"],
                poly_poles=[0, 2],
                correction=Correction.NONE,
                marg="full",
            )
            model.set_default("sigma_nl_par", 5.62)
            model.set_default("sigma_nl_perp", 3.01)
            fitter.add_model_and_dataset(model, data, name=names[counter])
            counter += 1

    for i, type in enumerate(types):
        for j in range(realisations[i]):
            print(i, type, j, realisations[i])
            data = CorrelationFunction_DESIMockChallenge_Post(
                isotropic=False, recon=recons[i], realisation=j, fit_poles=[0, 2], min_dist=52, max_dist=158, num_mocks=1000, type=type
            )
            model = CorrBeutler2017(
                recon=recon_types[i], isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.NONE, marg="full"
            )
            fitter.add_model_and_dataset(model, data, name=names[counter])
            counter += 1

            model = CorrBeutler2017(
                recon=recon_types[i],
                isotropic=False,
                fix_params=["om", "sigma_nl_par", "sigma_nl_perp"],
                poly_poles=[0, 2],
                correction=Correction.NONE,
                marg="full",
            )
            model.set_default("sigma_nl_par", 5.62)
            model.set_default("sigma_nl_perp", 3.01)
            fitter.add_model_and_dataset(model, data, name=names[counter])
            counter += 1

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        splits = ["Pk", "Xi"]
        ntotal = [4, 4]
        for spl, split in enumerate(splits):
            output = []
            counter = 0
            c = ChainConsumer()
            for posterior, weight, chain, evidence, model, data, extra in fitter.load():

                color = plt.colors.rgb2hex(cmap(float(counter) / (ntotal[spl] - 1)))

                if split not in extra["name"]:
                    continue

                model.set_data(data)
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
                df["weights"] = weight
                df.to_csv(pfn + "_" + extra["name"].replace(" ", "_") + "_chain.csv", index=False)
                c.add_chain(df, weights=weight, color=color, **extra)

                max_post = posterior.argmax()
                chi2 = -2 * posterior[max_post]

                params = model.get_param_dict(chain[max_post])
                for name, val in params.items():
                    model.set_default(name, val)

                if "Pk" in extra["name"]:

                    # Ensures we return the window convolved model
                    icov_m_w = model.data[0]["icov_m_w"]
                    model.data[0]["icov_m_w"][0] = None

                    ks = model.data[0]["ks"]
                    err = np.sqrt(np.diag(model.data[0]["cov"]))
                    mod, mod_odd, polymod, polymod_odd, _ = model.get_model(params, model.data[0], data_name=data[0]["name"])

                    if model.marg:
                        mask = data[0]["m_w_mask"]
                        mod_fit, mod_fit_odd = mod[mask], mod_odd[mask]

                        len_poly = (
                            len(model.data[0]["ks"]) if model.isotropic else len(model.data[0]["ks"]) * len(model.data[0]["fit_poles"])
                        )
                        polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros(
                            (np.shape(polymod)[0], len_poly)
                        )
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
                        chi2 = new_chi_squared

                    model.data[0]["icov_m_w"] = icov_m_w
                    dof = data[0]["pk"].shape[0] - 1 - len(df.columns)

                else:

                    ss = model.data[0]["dist"]
                    err = np.sqrt(np.diag(model.data[0]["cov"]))
                    mod, polymod = model.get_model(params, model.data[0])

                    if model.marg:

                        mod_fit = break_vector_and_get_blocks(mod, len(model.data[0]["poles"]), model.data[0]["fit_pole_indices"])
                        polymod_fit = np.empty((np.shape(polymod)[0], len(model.data[0]["fit_pole_indices"]) * len(model.data[0]["dist"])))
                        for n in range(np.shape(polymod)[0]):
                            polymod_fit[n] = break_vector_and_get_blocks(
                                polymod[n], np.shape(polymod)[1] / len(model.data[0]["dist"]), model.data[0]["fit_pole_indices"]
                            )
                        bband = model.get_ML_nuisance(
                            model.data[0]["xi"],
                            mod_fit,
                            np.zeros(mod_fit.shape),
                            polymod_fit,
                            np.zeros(polymod_fit.shape),
                            model.data[0]["icov"],
                            [None],
                        )
                        mod += bband @ polymod
                        mod_fit += bband @ polymod_fit

                        new_chi_squared = -0.5 * model.get_chi2_likelihood(
                            model.data[0]["xi"],
                            mod_fit,
                            np.zeros(mod_fit.shape),
                            model.data[0]["icov"],
                            [None],
                            num_mocks=model.data[0]["num_mocks"],
                            num_params=len(model.get_active_params()) + len(bband),
                        )
                        alphas = model.get_alphas(params["alpha"], params["epsilon"])
                        print(new_chi_squared, len(model.data[0]["xi"]) - len(model.get_active_params()) - len(bband), bband)
                        chi2 = new_chi_squared

                    dof = data[0]["xi"].shape[0] - 1 - len(df.columns)

                ps = chain[max_post, :]
                best_fit = {}
                for l, p in zip(model.get_labels(), ps):
                    best_fit[l] = p

                if "Fixed Sigma" in extra["name"]:
                    mean, cov = weighted_avg_and_cov(
                        df[
                            [
                                "$\\alpha_\\parallel$",
                                "$\\alpha_\\perp$",
                                "$\\Sigma_s$",
                                "$\\beta$",
                            ]
                        ],
                        weight,
                        axis=0,
                    )
                    mean = np.concatenate([mean, [5.62, 3.01]])
                else:
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

                c2 = ChainConsumer()
                c2.add_chain(df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]], weights=weight)
                corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
                if "Pk" in extra["name"]:
                    output.append(
                        f"{extra['name']:32s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3g}, {r_s:8.3f}, {chi2:7.3g}, {dof:4d}, {mean[4]:7.3g}, {mean[5]:7.3g}, {mean[2]:7.3g}, {mean[3]:7.3g}, {bband[0]:7.3g}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:7.3g}, {bband[4]:8.1f}, {bband[5]:8.1f}, {bband[6]:7.3g}"
                    )
                else:
                    output.append(
                        f"{extra['name']:32s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3g}, {r_s:8.3f}, {chi2:7.3g}, {dof:4d}, {mean[4]:7.3g}, {mean[5]:7.3g}, {mean[2]:7.3g}, {mean[3]:7.3g}, {bband[0]:7.3g}, {bband[4]:7.3g}, {bband[1]:7.3g}, {bband[2]:7.3g}, {bband[3]:7.3g}, {bband[5]:7.3g}, {bband[6]:7.3g}, {bband[7]:7.3g}"
                    )

                counter += 1

            c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, statistics="mean", legend_location=(0, -1))
            truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
            c.plotter.plot_summary(
                filename=[pfn + "_" + split + "_summary.pdf"],
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
                filename=[pfn + "_" + split + "_contour.pdf"],
                truth=truth,
                parameters=["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
            )
            c.plotter.plot(filename=[pfn + "_" + split + "_contour2.pdf"], truth=truth)
            c.analysis.get_latex_table(filename=pfn + "_" + split + "_params.txt")

            with open(pfn + "_" + split + "_BAO_fitting.Barry", "w") as f:
                if "Pk" in split:
                    f.write(
                        "#Name, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_2, a0_1, a0_0, a2_2, a2_1, a2_0\n"
                    )
                else:
                    f.write(
                        "#Name, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b0, b2, a0_2, a0_1, a0_0, a2_2, a2_1, a2_0\n"
                    )
                for l in output:
                    f.write(l + "\n")
