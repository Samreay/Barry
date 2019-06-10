import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    dataRecon = MockPowerSpectrum(name="Recon mean", recon=True, min_k=0.03, max_k=0.30)
    dataPrerecon = MockPowerSpectrum(name="Prerecon mean", recon=False, min_k=0.03, max_k=0.30)

    sampler = EnsembleSampler(temp_dir=dir_name)
    fitter = Fitter(dir_name)

    fitter.add_model_and_dataset(PowerSeo2016(recon=True), dataRecon, name="Seo Recon", linestyle="-", color="r")
    fitter.add_model_and_dataset(PowerDing2018(recon=True), dataRecon, name="Ding Recon", linestyle="-", color="lb")
    fitter.add_model_and_dataset(PowerBeutler2017(recon=True), dataRecon, name="Beutler Recon", linestyle="-", color="p")
    fitter.add_model_and_dataset(PowerSeo2016(recon=False), dataPrerecon, name="Seo Prerecon", linestyle="--", color="r")
    fitter.add_model_and_dataset(PowerDing2018(recon=False), dataPrerecon, name="Ding Prerecon", linestyle="--", color="lb")
    fitter.add_model_and_dataset(PowerBeutler2017(recon=False), dataPrerecon, name="Beutler Prerecon", linestyle="--", color="p")

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=20)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



