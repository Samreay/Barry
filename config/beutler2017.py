import sys
sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerBeutler2017
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    sampler = EnsembleSampler(temp_dir=dir_name, num_walkers=200)
    fitter = Fitter(dir_name)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        ls = "-" if r else "--"
        c = "p" if r else "o"
        d = MockPowerSpectrum(recon=r, min_k=0.03, max_k=0.30, reduce_cov_factor=31)
        fitter.add_model_and_dataset(PowerBeutler2017(recon=r), d, name=f"Beutler {t}", linestyle=ls, color=c)

    fitter.set_sampler(sampler)
    fitter.set_num_walkers(20)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)
        c.configure(shade=True, bins=30, legend_artists=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        c.plotter.plot_summary(filename=pfn + "_summary.png", errorbar=True, truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table())
        # c.plotter.plot_walks(filename=pfn + "_walks.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})



