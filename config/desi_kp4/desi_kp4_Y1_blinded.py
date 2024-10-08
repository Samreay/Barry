import sys

sys.path.append("..")
sys.path.append("../..")
from barry.samplers import NautilusSampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
import numpy as np
import scipy as sp
import pandas as pd
from barry.models.model import Correction
from barry.utils import weighted_avg_and_cov
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import pickle
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__, "/v2/")

    # Set up the Fitting class and Dynesty sampler with 250 live points.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = NautilusSampler(temp_dir=dir_name)

    colors = [mplc.cnames[color] for color in ["orange", "orangered", "firebrick", "lightskyblue", "steelblue", "seagreen", "black"]]

    tracers = {
        "LRG": [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
        "ELG_LOPnotQSO": [[0.8, 1.1], [1.1, 1.6]],
        "QSO": [[0.8, 2.1]],
        "BGS_BRIGHT-21.5": [[0.1, 0.4]],
    }
    reconsmooth = {"LRG": 10, "ELG_LOPnotQSO": 10, "QSO": 30, "BGS_BRIGHT-21.5": 15}
    sigma_nl_par = {
        "LRG": [
            [9.0, 6.0],
            [9.0, 6.0],
            [9.0, 6.0],
        ],
        "ELG_LOPnotQSO": [[8.5, 6.0], [8.5, 6.0]],
        "QSO": [[9.0, 6.0]],
        "BGS_BRIGHT-21.5": [[10.0, 8.0]],
    }
    sigma_nl_perp = {
        "LRG": [
            [4.5, 3.0],
            [4.5, 3.0],
            [4.5, 3.0],
        ],
        "ELG_LOPnotQSO": [[4.5, 3.0], [4.5, 3.0]],
        "QSO": [[3.5, 3.0]],
        "BGS_BRIGHT-21.5": [[6.5, 3.0]],
    }
    sigma_s = {
        "LRG": [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        "ELG_LOPnotQSO": [[2.0, 2.0], [2.0, 2.0]],
        "QSO": [[2.0, 2.0]],
        "BGS_BRIGHT-21.5": [[2.0, 2.0]],
    }

    allnames = []
    cap = "gccomb"
    ffa = "ffa"  # Flavour of fibre assignment. Can be "ffa" for fast fiber assign, or "complete"
    rpcut = False  # Whether or not to include the rpcut
    imaging = "default_FKP"  # What form of imaging systematics to use. Can be "default_FKP", "default_FKP_addSN", or "default_FKP_addRF"
    rp = f"{imaging}_rpcut2.5" if rpcut else f"{imaging}"
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for r, recon in enumerate([None, "sym"]):
                name = f"DESI_Y1_BLIND_v1_sm{reconsmooth[t]}_{t.lower()}_{cap}_z{zs[0]}-{zs[1]}_default_FKP_xi.pkl"
                print(name)
                dataset = CorrelationFunction_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_dist=50.0,
                    max_dist=151.0,
                    realisation="data",
                    reduce_cov_factor=1,
                    datafile=name,
                )
                model = CorrBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=dataset.fit_poles,
                    correction=Correction.NONE,
                    broadband_type="spline",
                    n_poly=[0, 2],
                )
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                model.set_default("beta", 0.4, min=0.1, max=0.7)
                model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                name = dataset.name + f" xi_spline"
                fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i])
                allnames.append(name)

                name = f"DESI_Y1_BLIND_v1_sm{reconsmooth[t]}_{t.lower()}_{cap}_z{zs[0]}-{zs[1]}_pk.pkl"
                print(name)
                dataset = PowerSpectrum_DESI_KP4(
                    recon=recon,
                    fit_poles=[0, 2],
                    min_k=0.02,
                    max_k=0.30,
                    realisation="data",
                    reduce_cov_factor=1,
                    datafile=name,
                )
                print(dataset.get_data()[0]["ks"])
                exit()
                model = PowerBeutler2017(
                    recon=dataset.recon,
                    isotropic=dataset.isotropic,
                    marg="full",
                    fix_params=["om"],
                    poly_poles=dataset.fit_poles,
                    correction=Correction.NONE,
                    broadband_type="spline",
                    n_poly=30,
                )
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=4.0)
                model.set_default("beta", 0.4, min=0.1, max=0.7)
                model.set_default("sigma_nl_par", sigma_nl_par[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                model.set_default("sigma_nl_perp", sigma_nl_perp[t][i][r], min=0.0, max=20.0, sigma=1.0, prior="gaussian")
                model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.kvals, model.pksmooth, model.pkratio = pktemplate.T

                name = dataset.name + f" pk_spline"
                fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i])
                allnames.append(name)

    # Submit all the job. We have quite a few (42), so we'll
    # only assign 1 walker (processor) to each. Note that this will only run if the
    # directory is empty (i.e., it won't overwrite existing chains)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    # Everything below here is for plotting the chains once they have been run. The should_plot()
    # function will check for the presence of chains and plot if it finds them on your laptop. On the HPC you can
    # also force this by passing in "plot" as the second argument when calling this code from the command line.
    if fitter.should_plot():
        import logging

        logging.info("Creating plots")
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

        # Set up a ChainConsumer instance. Plot the MAP for individual realisations and a contour for the mock average
        plotnames = [f"{t.lower()}_{zs[0]}_{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
        datanames = [f"{t.lower()}_{cap}_z{zs[0]}-{zs[1]}" for t in tracers for i, zs in enumerate(tracers[t])]
        print(datanames)
        c = [ChainConsumer() for i in range(len(datanames) * 2)]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the tracer bin, sigma bin and n_poly bin
            data_bin = datanames.index(extra["name"].split(" ")[5].lower())
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            poly_bin = 0 if "xi_spline" in extra["name"] else 1
            stats_bin = recon_bin * len(datanames) + data_bin
            pname = "xi" if poly_bin == 0 else "pk"
            # print(extra["name"], data_bin, recon_bin, poly_bin, stats_bin)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())

            # Compute alpha_par and alpha_perp for each point in the chain
            alpha_par, alpha_perp = model.get_alphas(df["$\\alpha$"].to_numpy(), df["$\\epsilon$"].to_numpy())
            df["$\\alpha_\\parallel$"] = alpha_par
            df["$\\alpha_\\perp$"] = alpha_perp
            df["$\\alpha_{ap}$"] = (1.0 + df["$\\epsilon$"].to_numpy()) ** 3
            newweight = np.where(
                np.logical_and(
                    np.logical_and(df["$\\alpha_\\parallel$"] >= 0.8, df["$\\alpha_\\parallel$"] <= 1.2),
                    np.logical_and(df["$\\alpha_\\perp$"] >= 0.8, df["$\\alpha_\\perp$"] <= 1.2),
                ),
                weight,
                0.0,
            )

            df["$d\\alpha_\\parallel$"] = 100.0 * (alpha_par - 1.0)
            df["$d\\alpha_\\perp$"] = 100.0 * (alpha_perp - 1.0)
            df["$d\\alpha_{ap}$"] = 100.0 * ((1.0 + df["$\\epsilon$"].to_numpy()) ** 3 - 1.0)
            df["$d\\alpha$"] = 100.0 * (df["$\\alpha$"] - 1.0)
            df["$d\\epsilon$"] = 100.0 * df["$\\epsilon$"]

            # Get the MAP point and set the model up at this point
            model.set_data(data)
            r_s = model.camb.get_data()["r_s"]
            max_post = posterior.argmax()
            params = df.loc[max_post]
            params_dict = model.get_param_dict(chain[max_post])
            for name, val in params_dict.items():
                model.set_default(name, val)

            # Get some useful properties of the fit, and plot the MAP model against the data
            plotname = f"{plotnames[data_bin]}_prerecon" if recon_bin == 0 else f"{plotnames[data_bin]}_postrecon"
            figname = "/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_{pname}_bestfit.png"
            new_chi_squared, dof, bband, mods, smooths = model.simple_plot(
                params_dict, display=False, figname=figname, title=plotname, c=colors[data_bin]
            )
            print(extra["name"], poly_bin, recon_bin, np.corrcoef(alpha_par, alpha_perp)[0, 1], new_chi_squared, dof)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            extra.pop("name", None)
            c[stats_bin].add_chain(df, weights=newweight, name=f"{pname}", plot_contour=True, plot_point=False, show_as_1d_prior=False)

            df["weight"] = newweight
            df.to_csv("/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_{pname}.dat", index=False, sep=" ")

        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    dataname = f"{t.lower()}_{cap}_z{zs[0]}-{zs[1]}"
                    data_bin = datanames.index(dataname.lower())
                    stats_bin = recon_bin * len(datanames) + data_bin

                    truth = {
                        "$\\alpha$": 1.0,
                        "$\\alpha_{ap}$": 1.0,
                        "$\\alpha_\\perp$": 1.0,
                        "$\\alpha_\\parallel$": 1.0,
                        "$\\Sigma_{nl,||}$": sigma_nl_par[t][i][recon_bin],
                        "$\\Sigma_{nl,\\perp}$": sigma_nl_perp[t][i][recon_bin],
                        "$\\Sigma_s$": sigma_s[t][i][recon_bin],
                    }

                    plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha_\\parallel$",
                            "$\\alpha_\\perp$",
                            "$\\Sigma_{nl,||}$",
                            "$\\Sigma_{nl,\\perp}$",
                            "$\\Sigma_s$",
                        ],
                    )
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour2.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha$",
                            "$\\alpha_{ap}$",
                            "$\\Sigma_{nl,||}$",
                            "$\\Sigma_{nl,\\perp}$",
                            "$\\Sigma_s$",
                        ],
                    )

                    print(
                        data_bin,
                        recon_bin,
                        c[stats_bin].analysis.get_latex_table(
                            parameters=["$\\alpha$", "$\\alpha_{ap}$", "$\\epsilon$", "$\\alpha_\\parallel$", "$\\alpha_\\perp$"]
                        ),
                    )
