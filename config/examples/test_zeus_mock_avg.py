import sys

sys.path.append("..")
sys.path.append("../..")
from barry.config import setup
from barry.fitter import Fitter
from barry.models.bao_power_Beutler2017 import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12
from barry.utils import plot_bestfit
from barry.samplers import ZeusSampler

# Run a quick test using dynesty to fit a mock mean.

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    data = PowerSpectrum_SDSS_DR12(isotropic=True, recon="iso")
    model = PowerBeutler2017(isotropic=data.isotropic, recon=data.recon, marg="full")

    sampler = ZeusSampler(temp_dir=dir_name)

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, data)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        posterior, weight, chain, evidence, model, data, extra = fitter.load()[0]
        chi2, dof, bband, mods, smooths = plot_bestfit(posterior, chain, model, title=extra["name"], figname=pfn + "_bestfit.pdf")

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=model.get_labels())
        c.plotter.plot(filename=pfn + "_contour.pdf")
