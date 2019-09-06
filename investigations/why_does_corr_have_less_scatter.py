import logging
from barry.cosmology.pk2xi import PowerToCorrelationGauss
from barry.models import PowerBeutler2017, PowerDing2018, PowerSeo2016
from barry.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC, PowerSpectrum_SDSS_DR12_Z061_NGC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    realisation = 599
    recon = True

    beutler = PowerBeutler2017(recon=recon)
    beutler.set_default("sigma_nl", 4.894)
    beutler.set_fix_params(["om", "sigma_nl"])

    models_pk = [beutler, PowerDing2018(recon=recon), PowerSeo2016(recon=True)]
    models_xi = [None]
    labels = ["Beutler 2017", "Ding 2018", "Seo 2016"]

    dataset_pk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon, realisation=realisation)
    dataset_xi = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=recon, realisation=realisation)

    data_pk = dataset_pk.get_data()
    data_xi = dataset_xi.get_data()

    fig, axes = plt.subplots(nrows=2)

    for l, pk_model in zip(labels, models_pk):

        # First comparison - the actual recon data
        pk_model.set_data(data_pk)
        p, minv = pk_model.optimize(niter=30, maxiter=1000)

        ks = pk_model.camb.ks
        print(l, p)
        # pk_model.plot(p)
        pks_1 = pk_model.compute_power_spectrum(ks, p)

        for n in ["a1", "a2", "a3", "a4", "a5"]:
            p[n] = 0
        pks_2 = pk_model.compute_power_spectrum(ks, p)

        pk2xi = PowerToCorrelationGauss(ks)
        ss = data_xi[0]["dist"]
        xi = pk2xi.pk2xi(ks, pks_2, ss)

        m = (ks >= 0.03) & (ks <= 0.3)
        axes[0].plot(ks[m], ks[m] * pks_1[m], label=l)
        axes[1].plot(ss, ss ** 2 * xi)
    axes[0].legend()
    plt.show()

    # Findings: Its not about the similarity of the model in xi(s), its just the data ranges used
