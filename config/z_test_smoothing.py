import sys

sys.path.append("..")
from barry.config import setup
from barry.models import PowerBeutler2017
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.samplers import DynestySampler
from barry.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    r = True
    models = [PowerBeutler2017(recon=r, smooth_type="hinton2017", name="Hinton2017"), PowerBeutler2017(recon=r, smooth_type="eh1998", name="EH1998")]
    data = PowerSpectrum_SDSS_DR12_Z061_NGC(name="Recon mean", recon=r, min_k=0.02, max_k=0.30)
    sampler = DynestySampler(temp_dir=dir_name)

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(models[0], data, name="Hinton2017")
    fitter.add_model_and_dataset(models[1], data, name="EH1998")
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        pks = {}
        for posterior, weight, chain, model, data, extra in fitter.load():
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), **extra)

            # params = dict([(p.name, v) for p, v in zip(model.get_active_params(), chain[posterior.argmax(), :])])
            # params["om"] = 0.3121
            # model.set_data(datas[0].get_data())
            # key = f"{model.name}, alpha={params['alpha']:0.4f}"
            # pks[key] = model.get_model(datas[0].get_data(), params)

        c.configure(shade=True, bins=0.7)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, "$\\alpha$": 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))
