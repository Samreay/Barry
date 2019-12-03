import logging

from barry.datasets.dataset import MultiDataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum


class PowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.61    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__(
            "sdss_dr12_z061_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )


class PowerSpectrum_SDSS_DR12_Z051_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__(
            "sdss_dr12_z051_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )


class PowerSpectrum_SDSS_DR12_Z051_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__(
            "sdss_dr12_z051_pk_sgc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )


class PowerSpectrum_SDSS_DR12_Z051(MultiDataset):
    """ Power spectrum for SDSS BOSS DR12 sample for combined NGC and SGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        ngc = PowerSpectrum_SDSS_DR12_Z051_NGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )
        sgc = PowerSpectrum_SDSS_DR12_Z051_SGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )
        super().__init__(name, [ngc, sgc])


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
        datasets = [PowerSpectrum_SDSS_DR12_Z061_NGC(recon=r)]
        for dataset in datasets:
            data = dataset.get_data()
            plt.errorbar(data[0]["ks"], data[0]["ks"] * data[0]["pk"], yerr=data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"])), fmt="o", c="k", zorder=1)
            for i, realisation in enumerate(np.random.randint(999, size=10)):
                dataset.set_realisation(realisation)
                data = dataset.get_data()
                plt.errorbar(data[0]["ks"], data[0]["ks"] * data[0]["pk"], fmt="-", c=cmap(i), zorder=0)
            plt.xlabel(r"$k$")
            plt.ylabel(r"$k\,P(k)$")
            plt.title(dataset.name + " " + t)

            plt.show()
