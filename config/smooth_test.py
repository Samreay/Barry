import sys
sys.path.append("..")
from barry.setup import setup
from barry.framework.models import PowerBeutler2017
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.framework.fitter import Fitter

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    r = True
    models = [
        PowerBeutler2017(recon=r, smooth_type="hinton2017", name="Hinton2017"),
        PowerBeutler2017(recon=r, smooth_type="eh1998", name="EH1998")
    ]
    datas = [MockPowerSpectrum(name="Recon mean", recon=r, min_k=0.03, max_k=0.30)]
    sampler = EnsembleSampler(num_steps=2500, num_burn=500, temp_dir=dir_name, save_interval=30)

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_data(*datas)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(50)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        pks = {}
        for posterior, weight, chain, model, data in fitter.load():
            name = f"{model.get_name()} {data.get_name()}"
            linestyle = "--" if "FitOm" in name else "-"
            c.add_chain(chain, weights=weight, parameters=model.get_labels(), name=name, linestyle=linestyle)

            params = dict([(p.name, v) for p, v in zip(model.get_active_params(), chain[posterior.argmax(), :])])
            params["om"] = 0.3121
            model.set_data(datas[0].get_data())
            key = f"{model.name}, alpha={params['alpha']:0.4f}"
            pks[key] = model.get_model(datas[0].get_data(), params)
            print(params)

        import matplotlib.pyplot as plt
        for n, v in pks.items():
            plt.plot(datas[0].get_data()["ks"], v, label=n)
        plt.plot(datas[0].get_data()["ks"], datas[0].get_data()["pk"], label="Data", c='k', ls='--')
        plt.legend()
        plt.show()
        print(datas[0].get_data()["ks"])

        c.configure(shade=True)
        c.plotter.plot(filename=pfn + "_contour.png", truth={"$\\Omega_m$": 0.3121, '$\\alpha$': 1.0})
        with open(pfn + "_params.txt", "w") as f:
            f.write(c.analysis.get_latex_table(transpose=True))



