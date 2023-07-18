import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")
from barry.config import setup
from barry.fitter import Fitter
from barry.models.bao_power_Beutler2017 import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12
from barry.utils import plot_bestfit
from barry.samplers import GridSearch

# Run a quick test using dynesty to fit a mock mean.

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    data = PowerSpectrum_SDSS_DR12(isotropic=True, recon="iso")
    model = PowerBeutler2017(isotropic=data.isotropic, recon=data.recon, marg="full")

    sampler = GridSearch(temp_dir=dir_name)

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, data)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    if fitter.should_plot():

        posterior, weight, chain, evidence, chi2, model, data, extra = fitter.load()[0]

        print(np.shape(chain), model.get_labels())
        df = pd.DataFrame(chain, columns=model.get_labels())

        # Get the MAP point and set the model up at this point
        model.set_data(data)
        r_s = model.camb.get_data()["r_s"]
        max_post = posterior.argmax()
        params = df.loc[max_post]
        params_dict = model.get_param_dict(chain[max_post])
        for name, val in params_dict.items():
            model.set_default(name, val)

        chi2_bb, dof, bband, mods, smooths = plot_bestfit(posterior, chain, model, title=extra["name"], figname=pfn + "_bestfit.pdf")

        plt.errorbar(df["$\\alpha$"], chi2, marker="None", ls="-", color="k")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\chi^2$")
        plt.xlim(0.7, 1.3)
        plt.show()
