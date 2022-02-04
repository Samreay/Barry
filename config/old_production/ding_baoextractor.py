import sys

sys.path.append("..")
from barry.cosmology.camb_generator import getCambGenerator
from barry.postprocessing import BAOExtractor
from barry.config import setup
from barry.utils import weighted_avg_and_std, get_model_comparison_dataframe
from barry.models import PowerDing2018, PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter
import numpy as np
import pandas as pd


# Check if B17 and D18 results change if we apply the BAO extractor technique.
# Spoiler: They do not.
if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)
    fitter = Fitter(dir_name, remove_output=False)

    c = getCambGenerator()
    r_s = c.get_data()["r_s"]
    p = BAOExtractor(r_s)

    sampler = DynestySampler(temp_dir=dir_name, nlive=200)

    for r in [True]:  # , False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"

        d = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, realisation=0)
        de = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r, postprocess=p, realisation=0)

        ding = PowerDing2018(recon=r)
        beutler = PowerBeutler2017(recon=r)
        sigma_nl = 6.0
        beutler.set_default("sigma_nl", sigma_nl)
        beutler.set_fix_params(["om", "sigma_nl"])

        beutler_extracted = PowerBeutler2017(recon=r, postprocess=p)
        beutler_extracted.set_default("sigma_nl", sigma_nl)
        beutler_extracted.set_fix_params(["om", "sigma_nl"])
        ding_extracted = PowerDing2018(recon=r, postprocess=p)

        for i in range(999):
            d.set_realisation(i)
            de.set_realisation(i)
            fitter.add_model_and_dataset(ding, d, name=f"D18", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler, d, name=f"B17", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(ding_extracted, de, name=f"D18 + Extractor", linestyle=ls, color="p", realisation=i)
            fitter.add_model_and_dataset(beutler_extracted, de, name=f"B17 + Extractor", linestyle=ls, color="p", realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(700)
    if not fitter.should_plot():
        fitter.fit(file)

    if fitter.should_plot():
        import matplotlib.pyplot as plt
        import logging

        logging.info("Creating plots")

        model_results, summary = get_model_comparison_dataframe(fitter)

        # Define colour scheme and plt defaults
        c4 = ["#262232", "#116A71", "#48AB75", "#D1E05B"]
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Create scatter pull plot. Figure 6 in the paper at the moment.
        if True:
            fig, axes = plt.subplots(nrows=2, figsize=(5, 4), sharex=True, gridspec_kw={"hspace": 0.0})
            pairs = [("B17", "B17 + Extractor"), ("D18", "D18 + Extractor")]

            for pair, ax, index in zip(pairs, axes, [0, 2]):
                combined_df = pd.merge(model_results[pair[0]], model_results[pair[1]], on="realisation", suffixes=("_original", "_extracted"))

                data = combined_df[["avg_original", "avg_extracted"]].values
                corr = np.corrcoef(data.T)[1, 0]
                print("corr is ", corr)

                step = 9
                alpha_diff = combined_df["avg_original"] - combined_df["avg_extracted"]
                print(np.mean(alpha_diff), np.std(alpha_diff))
                err = np.sqrt(
                    combined_df["std_original"] ** 2 + combined_df["std_extracted"] ** 2 - 2 * corr * combined_df["std_extracted"] * combined_df["std_original"]
                )
                chi2 = (np.abs(alpha_diff) / err).sum() / alpha_diff.size
                print("chi2 is ", chi2)
                x = combined_df["realisation"][::step]
                ax.errorbar(x, alpha_diff[::step], c=c4[index], yerr=err[::step], fmt="o", elinewidth=0.5, ms=2)
                ax.axhline(0, c="k", lw=1, ls="--")
                ax.set_ylabel(r"$\Delta\alpha$ for " + pair[0], fontsize=14)
                ax.set_ylim(-0.012, 0.012)
            axes[1].set_xlabel("Realisation", fontsize=14)
            plt.savefig(pfn + "_alphadiff.pdf", bbox_inches="tight", dpi=300, transparent=True)
            plt.savefig(pfn + "_alphadiff.png", bbox_inches="tight", dpi=300, transparent=True)
