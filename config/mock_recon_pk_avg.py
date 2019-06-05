import sys


sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerNoda2019, PowerSeo2016, PowerBeutler2017, PowerDing2018
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    r = True

    models = [
        # PowerNoda2019(recon=r),
        PowerSeo2016(recon=r),
        PowerDing2018(recon=r),
        PowerBeutler2017(recon=r)
    ]

    datas = [MockPowerSpectrum(name="Recon mean", recon=r, min_k=0.03, max_k=0.30)]

    sampler = EnsembleSampler(num_steps=1500, num_burn=500, temp_dir=dir_name, save_interval=30)

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_data(*datas)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        for posterior, weight, chain, model, data in fitter.load():
            name = f"{model.get_name()} {data.get_name()}"
            linestyle = "--" if "FitOm" in name else "-"
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)
        c.configure(shade=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



