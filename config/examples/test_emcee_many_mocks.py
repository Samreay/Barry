import sys

sys.path.append("..")
sys.path.append("../..")
from barry.config import setup
from barry.fitter import Fitter
from barry.models.bao_power_Beutler2017 import PowerBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12
from barry.utils import get_model_comparison_dataframe
from barry.samplers import EnsembleSampler

# Run a not so quick test submitting 128 jobs each with a different mock realisation.
# If run on NERSC, this will actually submit 128/num_per_task jobs, and each job will
# do num_per_task.

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    data = PowerSpectrum_SDSS_DR12(isotropic=True, recon="iso")
    model = PowerBeutler2017(isotropic=data.isotropic, recon=data.recon, marg="full")

    sampler = EnsembleSampler(num_walkers=16, num_steps=5000, num_burn=300, temp_dir=dir_name)

    fitter = Fitter(dir_name)
    for i in range(128):
        data.set_realisation(i)
        fitter.add_model_and_dataset(model, data, realisation=i)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.set_num_concurrent(128)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        posterior, weight, chain, evidence, model, data, extra = fitter.load()[0]
        model_results, summary = get_model_comparison_dataframe(fitter)

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=model.get_labels())
        c.plotter.plot(filename=pfn + "_contour.pdf")
