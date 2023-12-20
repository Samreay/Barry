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
from chainconsumer import ChainConsumer

# Config file to fit the abacus cutsky mock means for sigmas
if __name__ == "__main__":

    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

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
                dataset = CorrelationFunction_DESI_KP4(
                    isotropic=True,
                    recon=recon,
                    fit_poles=[0],
                    min_dist=50.0,
                    max_dist=150.0,
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
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=8.0)
                model.set_default(
                    "sigma_nl",
                    np.sqrt((sigma_nl_par[t][i][r] ** 2 + 2.0 * sigma_nl_perp[t][i][r] ** 2) / 3.0),
                    min=0.0,
                    max=20.0,
                    sigma=2.0,
                    prior="gaussian",
                )
                model.set_default("sigma_s", sigma_s[t][i][r], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                # Load in a pre-existing BAO template
                pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
                model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

                name = dataset.name + f" xi_spline"
                fitter.add_model_and_dataset(model, dataset, name=name, color=colors[i])
                allnames.append(name)

                name = f"DESI_Y1_BLIND_v1_sm{reconsmooth[t]}_{t.lower()}_{cap}_z{zs[0]}-{zs[1]}_pk.pkl"
                dataset = PowerSpectrum_DESI_KP4(
                    isotropic=True,
                    recon=recon,
                    fit_poles=[0],
                    min_k=0.02,
                    max_k=0.30,
                    realisation="data",
                    reduce_cov_factor=1,
                    datafile=name,
                )
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
                model.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=8.0)
                model.set_default(
                    "sigma_nl",
                    np.sqrt((sigma_nl_par[t][i][r] ** 2 + 2.0 * sigma_nl_perp[t][i][r] ** 2) / 3.0),
                    min=0.0,
                    max=20.0,
                    sigma=2.0,
                    prior="gaussian",
                )
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
        c = [ChainConsumer() for i in range(len(datanames) * 2)]
        stats = [[[], []] for _ in range(len(datanames))]
        for posterior, weight, chain, evidence, model, data, extra in fitter.load():

            # Get the tracer bin, sigma bin and n_poly bin
            data_bin = datanames.index(extra["name"].split(" ")[5].lower())
            recon_bin = 0 if "Prerecon" in extra["name"] else 1
            poly_bin = 0 if "xi_spline" in extra["name"] else 1
            stats_bin = recon_bin * len(datanames) + data_bin
            pname = "xi" if poly_bin == 0 else "pk"
            print(extra["name"], data_bin, recon_bin, poly_bin, stats_bin)

            # Store the chain in a dictionary with parameter names
            df = pd.DataFrame(chain, columns=model.get_labels())
            df["$d\\alpha$"] = 100.0 * (df["$\\alpha$"] - 1.0)

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
            print(extra["name"], poly_bin, recon_bin, new_chi_squared, dof)

            # Add the chain or MAP to the Chainconsumer plots
            extra.pop("realisation", None)
            extra.pop("name", None)
            c[stats_bin].add_chain(df, weights=weight, name=f"{pname}", plot_contour=True, plot_point=False, show_as_1d_prior=False)

            # Compute some summary statistics and add them to a dictionary
            mean, cov = weighted_avg_and_cov(
                df[
                    [
                        "$\\alpha$",
                        "$\\Sigma_{nl}$",
                        "$\\Sigma_s$",
                    ]
                ],
                weight,
                axis=0,
            )

            stats[data_bin][recon_bin].append(
                [
                    mean[0],
                    np.sqrt(cov[0, 0]),
                    new_chi_squared,
                ]
            )

        for t in tracers:
            for i, zs in enumerate(tracers[t]):
                for recon_bin in range(2):
                    dataname = f"{t.lower()}_{cap}_z{zs[0]}-{zs[1]}"
                    data_bin = datanames.index(dataname.lower())
                    stats_bin = recon_bin * len(datanames) + data_bin

                    truth = {
                        "$\\alpha$": 1.0,
                        "$\\Sigma_{nl}$": np.sqrt((sigma_nl_par[t][i][recon_bin] ** 2 + 2.0 * sigma_nl_perp[t][i][recon_bin] ** 2) / 3.0),
                        "$\\Sigma_s$": sigma_s[t][i][recon_bin],
                    }

                    plotname = f"{dataname}_prerecon" if recon_bin == 0 else f"{dataname}_postrecon"
                    c[stats_bin].plotter.plot(
                        filename=["/".join(pfn.split("/")[:-1]) + "/" + plotname + f"_contour2.png"],
                        truth=truth,
                        parameters=[
                            "$\\alpha$",
                            "$\\Sigma_{nl}$",
                            "$\\Sigma_s$",
                        ],
                    )

                    print(
                        data_bin,
                        recon_bin,
                        c[stats_bin].analysis.get_latex_table(parameters=["$\\alpha$"]),
                    )
