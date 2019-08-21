import logging

from barry.framework.cosmology.pk2xi import PowerToCorrelationGauss
from barry.framework.models import CorrBeutler2017, CorrDing2018, CorrSeo2016, PowerBeutler2017, PowerDing2018, PowerSeo2016
from barry.framework.datasets import CorrelationFunction_SDSS_DR12_Z061_NGC, PowerSpectrum_SDSS_DR12_Z061_NGC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    realisation = 599
    recon = True

    models_pk = [PowerBeutler2017(recon=recon)]
    models_xi = [CorrBeutler2017()]
    labels = ["Beutler 2017"]

    dataset_pk = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon, realisation=realisation)
    dataset_xi = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=recon, realisation=realisation)

    data_pk = dataset_pk.get_data()
    data_xi = dataset_xi.get_data()

    for l, pk_model, xi_model in zip(labels, models_pk, models_xi):

        # First comparison - the actual recon data
        pk_model.set_data(data_pk)
        p, minv = pk_model.optimize(maxiter=500)

        ks = pk_model.camb.ks
        print(p)
        pk_model.plot(p)
        pks = pk_model.compute_power_spectrum(ks, p)

        pk2xi = PowerToCorrelationGauss(ks)
        ss = data_xi[0]["dist"]
        xi = pk2xi.pk2xi(ks, pks, ss)

        fig, axes = plt.subplots(nrows=2)

        axes[0].plot(ks, ks * pks)
        axes[0].set_xlim(0.02, 0.3)
        axes[0].set_ylim(700, 2000)
        axes[1].plot(ss, ss ** 2 * xi)
        plt.show()
