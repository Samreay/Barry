import sys

from chainconsumer import ChainConsumer

sys.path.append("..")
from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESILightcone_Mocks_Recon
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

    sampler = DynestySampler(temp_dir=dir_name, nlive=100)

    names = ["PreRecon", "PostRecon Julian RecIso", "PostRecon Julian RecSym", "PostRecon Martin RecIso", "PostRecon Martin RecSym"]
    # colors = ["#CAF270", "#CAF270", "#4AB482", "#1A6E73", "#232C3B"]
    colors = ["#CAF270", "#66C57F", "#219180", "#205C68", "#232C3B"]

    types = ["julian_reciso", "julian_reciso", "julian_recsym", "martin_reciso", "martin_recsym"]
    recons = [False, True, True, True, True]
    recon_types = ["None", "iso", "ani", "iso", "ani"]
    realisations = ["data", "data", "data", "data", "data"]

    for i, (name, type, recon, recon_type, realisation, color) in enumerate(zip(names, types, recons, recon_types, realisations, colors)):
        print(i, name, type, recon, recon_type, realisation)
        data = PowerSpectrum_DESILightcone_Mocks_Recon(
            isotropic=False, recon=recon, realisation=realisation, fit_poles=[0, 2], min_k=0.02, max_k=0.30, type=type
        )
        model = PowerBeutler2017(
            recon=recon_type, isotropic=False, fix_params=["om"], poly_poles=[0, 2], correction=Correction.NONE, marg="full"
        )
        fitter.add_model_and_dataset(model, data, name=name, color=color)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(5)
    fitter.fit(file)

    # Everything below is nasty plotting code ###########################################################
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")

        from chainconsumer import ChainConsumer

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        fig1, axes1 = plt.subplots(figsize=(5, 8), nrows=len(names), sharex=True, gridspec_kw={"hspace": 0.08})
        fig2, axes2 = plt.subplots(figsize=(5, 8), nrows=len(names), sharex=True, gridspec_kw={"hspace": 0.08})
        labels = [
            r"$k \times P(k)\,(h^{-2}\,\mathrm{Mpc^{2}})$",
            r"$k^{2} \times (P(k) - P_{\mathrm{smooth}}(k))\,(h^{-1}\,\mathrm{Mpc})$",
        ]
        for fig, label in zip([fig1, fig2], labels):
            ax = fig.add_subplot(111, frameon=False)
            ax.set_ylabel(label)
            ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$")
            ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

        output = []
        counter = 0
        c = ChainConsumer()
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # if "PreRecon" in extra["name"]:
            #    continue

            print(extra["name"])
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]

            df = pd.DataFrame(chain, columns=model.get_labels())
            alpha = df["$\\alpha$"].to_numpy()
            epsilon = df["$\\epsilon$"].to_numpy()
            print(model, np.shape(alpha), np.shape(epsilon))
            alpha_par, alpha_perp = model.get_alphas(alpha, epsilon)
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp

            extra.pop("realisation", None)
            c.add_chain(df, weights=weight, **extra)

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
            smooth, smooth_odd, polysmooth, polysmooth_odd, _ = model.get_model(
                params, model.data[0], data_name=data[0]["name"], smooth=True
            )

            if model.marg:
                mask = data[0]["m_w_mask"]
                mod_fit, mod_fit_odd = mod[mask], mod_odd[mask]
                smooth_fit, smooth_fit_odd = smooth[mask], smooth_odd[mask]

                len_poly = len(model.data[0]["ks"]) if model.isotropic else len(model.data[0]["ks"]) * len(model.data[0]["fit_poles"])
                polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
                for nn in range(np.shape(polymod)[0]):
                    polymod_fit[nn], polymod_fit_odd[nn] = polymod[nn, mask], polymod_odd[nn, mask]
                polysmooth_fit, polysmooth_fit_odd = np.empty((np.shape(polysmooth)[0], len_poly)), np.zeros(
                    (np.shape(polysmooth)[0], len_poly)
                )
                for nn in range(np.shape(polysmooth)[0]):
                    polysmooth_fit[nn], polysmooth_fit_odd[nn] = polysmooth[nn, mask], polysmooth_odd[nn, mask]

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
                smooth += smooth_odd + bband @ (polysmooth + polysmooth_odd)
                smooth_fit += smooth_fit_odd + bband @ (polysmooth_fit + polysmooth_fit_odd)

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

            # Add the data and model to the plot
            print(len(err), len(ks), model.data[0]["fit_pole_indices"])
            if len(err) > len(ks):
                assert len(err) % len(ks) == 0, f"Cannot split your data - have {len(err)} points and {len(ks)} modes"
            errs = [[col for col in err.reshape((-1, len(ks)))][ind] for ind in model.data[0]["fit_pole_indices"]]
            mods = [col for col in mod_fit.reshape((-1, len(ks)))]
            smooths = [col for col in smooth_fit.reshape((-1, len(ks)))]
            print(len(mods), len(smooths), len(errs))

            titles = [f"pk{n}" for n in model.data[0]["fit_poles"]]

            ax1 = fig1.add_subplot(axes1[counter])
            axes = fig2.add_subplot(axes2[counter])
            axes.spines["top"].set_color("none")
            axes.spines["bottom"].set_color("none")
            axes.spines["left"].set_color("none")
            axes.spines["right"].set_color("none")
            axes.tick_params(axis="both", which="both", labelcolor="none", top=False, bottom=False, left=False, right=False)

            mfcs = ["#666666", "w"]
            lines = ["-", "--"]
            inner = gridspec.GridSpecFromSubplotSpec(1, len(titles), subplot_spec=axes2[counter], wspace=0.08)
            for i, (inn, err, mod, smooth, title, line, mfc) in enumerate(zip(inner, errs, mods, smooths, titles, lines, mfcs)):

                ax1.errorbar(ks, ks * data[0][title], yerr=ks * err, fmt="o", ms=4, c="#666666", mfc=mfc)
                ax1.plot(ks, ks * mod, c=extra["color"], ls=line)
                if counter != (len(names) - 1):
                    ax1.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax1.annotate(extra["name"], xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top")

                ax2 = fig2.add_subplot(inn)
                ax2.errorbar(ks, ks ** 2 * (data[0][title] - smooth), yerr=ks ** 2 * err, fmt="o", ms=4, c="#666666")
                ax2.plot(ks, ks ** 2 * (mod - smooth), c=extra["color"])
                ax2.set_ylim(-4.0, 4.0)
                if counter == 0:
                    if i == 0:
                        ax2.set_title(r"$P_{0}(k)$")
                    elif i == 1:
                        ax2.set_title(r"$P_{2}(k)$")
                if counter != (len(names) - 1):
                    ax2.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)

                if i != 0:
                    ax2.tick_params(axis="y", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    ax2.annotate(extra["name"], xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top")

            counter += 1

            # Compute some summary statistics
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

            c2 = ChainConsumer()
            c2.add_chain(df[["$\\alpha_\\parallel$", "$\\alpha_\\perp$"]], weights=weight)
            corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
            output.append(
                f"{extra['name']:32s}, {mean[0]:6.4f}, {mean[1]:6.4f}, {np.sqrt(cov[0,0]):6.4f}, {np.sqrt(cov[1,1]):6.4f}, {corr:7.3f}, {r_s:7.3f}, {chi2:7.3f}, {dof:4d}, {mean[4]:7.3f}, {mean[5]:7.3f}, {mean[2]:7.3f}, {mean[3]:7.3f}, {bband[0]:7.3f}, {bband[1]:8.1f}, {bband[2]:8.1f}, {bband[3]:8.1f}, {bband[4]:7.3f}, {bband[5]:7.3f}, {bband[6]:8.1f}, {bband[7]:8.1f}, {bband[8]:8.1f}, {bband[9]:7.3f}, {bband[10]:7.3f}"
            )

        # Output all the figures
        fig1.savefig(pfn + "_bestfits.png", bbox_inches="tight", dpi=300, transparent=False)
        fig2.savefig(pfn + "_bestfits_2.png", bbox_inches="tight", dpi=300, transparent=False)

        c.configure(shade=True, bins=20, legend_artists=True, max_ticks=4, statistics="mean", legend_location=(0, -1))
        truth = {"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0, "$\\epsilon$": 0, "$\\alpha_\\perp$": 1.0, "$\\alpha_\\parallel$": 1.0}
        c.plotter.plot_summary(
            filename=[pfn + "_summary.png"],
            errorbar=True,
            truth=truth,
            parameters=["$\\alpha_\\parallel$", "$\\alpha_\\perp$", "$\\alpha$", "$\\epsilon$"],
            extents={
                "$\\alpha_\\parallel$": [0.97, 1.06],
                "$\\alpha_\\perp$": [0.98, 1.05],
                "$\\alpha$": [0.98, 1.06],
                "$\\epsilon$": [-0.015, 0.015],
            },
        )
        c.plotter.plot(
            filename=[pfn + "_contour.pdf"],
            truth=truth,
            parameters=["$\\alpha$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"],
            # extents={
            #    "$\\alpha_\\parallel$": [0.92, 1.10],
            #    "$\\alpha_\\perp$": [0.95, 1.08],
            #    "$\\alpha$": [0.95, 1.08],
            #    "$\\epsilon$": [-0.04, 0.04],
            # },
        )
        c.plotter.plot(
            filename=[pfn + "_contour2.png"],
            truth=truth,
            parameters=[
                "$\\alpha_\\parallel$",
                "$\\alpha_\\perp$",
                "$\\Sigma_s$",
                "$\\beta$",
                "$\\Sigma_{nl,||}$",
                "$\\Sigma_{nl,\\perp}$",
            ],
        )
        c.analysis.get_latex_table(filename=pfn + "_params.txt")

        with open(pfn + "_BAO_fitting.Barry", "w") as f:
            f.write(
                "#Name, best_fit_alpha_par, best_fit_alpha_perp, sigma_alpha_par, sigma_alpha_perp, corr_alpha_par_perp, rd_of_template, bf_chi2, dof, sigma_nl_par, sigma_nl_per, sigma_fog, beta, b, a0_1, a0_2, a0_3, a0_4, a0_5, a2_1, a2_2, a2_3, a2_4, a2_5, a4_1, a4_2, a4_3, a4_4, a4_5\n"
            )
            for l in output:
                f.write(l + "\n")
