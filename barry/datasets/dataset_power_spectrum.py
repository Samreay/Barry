import logging
from barry.datasets.dataset import MultiDataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum


class PowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum):
    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True,
                 min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__("sdss_dr12_ngc_pk_zbin0p61.pkl", name=name, min_k=min_k,
                         max_k=max_k, step_size=step_size, recon=recon,
                         reduce_cov_factor=reduce_cov_factor, postprocess=postprocess,
                         realisation=realisation, fake_diag=fake_diag)


class PowerSpectrum_SDSS_DR12_Z051_NGC(PowerSpectrum):
    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__("sdss_dr12_ngc_pk_zbin0p51.pkl", name=name, min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, fake_diag=fake_diag)


class PowerSpectrum_SDSS_DR12_Z051_SGC(PowerSpectrum):
    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        super().__init__("sdss_dr12_sgc_pk_zbin0p51.pkl", name=name, min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, fake_diag=fake_diag)


class PowerSpectrum_SDSS_DR12_Z051(MultiDataset):
    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        ngc = PowerSpectrum_SDSS_DR12_Z051_NGC(min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, fake_diag=fake_diag)
        sgc = PowerSpectrum_SDSS_DR12_Z051_NGC(min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, fake_diag=fake_diag)
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_SDSS_DR7_Z015(PowerSpectrum):
    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=5, postprocess=None):
        super().__init__("sdss_dr7_pk.pkl", name=name, min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, fake_diag=fake_diag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    # Some basic checks for data we expect to be there
    dataset = PowerSpectrum_SDSS_DR12_Z051_NGC(recon=False)
    # print(dataset.all_data)
    data = dataset.get_data()
    # print(data["ks"])
    #
    import matplotlib.pyplot as plt
    import numpy as np
    plt.errorbar(data["ks"], data["ks"]*data["pk"], yerr=data["ks"]*np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()


    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=1)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=2, recon=False)
    # MockAveragePowerSpectrum(step_size=5, recon=False)
