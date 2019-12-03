import logging

from barry.datasets.dataset_correlation_function_abc import CorrelationFunction


class CorrelationFunction_SDSS_DR7_Z015_MGS(CorrelationFunction):
    """ Correlation function for SDSS MGS DR7 sample with mean redshift z = 0.15    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        super().__init__(
            "sdss_dr7_corr.pkl", name=name, min_dist=min_dist, max_dist=max_dist, recon=recon, reduce_cov_factor=reduce_cov_factor, realisation=realisation
        )


class CorrelationFunction_SDSS_DR12_Z061_NGC(CorrelationFunction):
    """ Correlation function for SDSS BOSS DR12 sample for the NGC with mean redshift z = 0.61    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        super().__init__(
            "sdss_dr12_ngc_corr_zbin0p61.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            realisation=realisation,
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Make plots of the different datasets we expect to be there; the mean and errors, and 10 random realisations
    nrealisations = 10
    cmap = plt.cm.get_cmap("viridis", nrealisations)

    for r in [True, False]:
        t = "Recon" if r else "Prerecon"
        datasets = [CorrelationFunction_SDSS_DR7_Z015_MGS(recon=r), CorrelationFunction_SDSS_DR12_Z061_NGC(recon=r)]
        for dataset in datasets:
            data = dataset.get_data()
            plt.errorbar(
                data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi0"], yerr=data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"])), fmt="o", c="k", zorder=1
            )
            for i, realisation in enumerate(np.random.randint(999, size=10)):
                dataset.set_realisation(realisation)
                data = dataset.get_data()
                plt.errorbar(data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi0"], fmt="-", c=cmap(i), zorder=0)
            plt.xlabel(r"$s$")
            plt.ylabel(r"$s^{2}\xi(s)$")
            plt.title(dataset.name)

            plt.show()
