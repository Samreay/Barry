import logging
from scipy.interpolate import splev, splrep
import numpy as np

from barry.cosmology import pk2xi
from barry.cosmology.camb_generator import getCambGenerator
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC, CorrelationFunction_SDSS_DR12_Z061_NGC


class DummyPowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum_SDSS_DR12_Z061_NGC):
    """ Dummy power spectrum.

    Uses CAMB's linear power spectrum and faked uncertainty. Utilised the SDSS DR12 window function, with option
    to make a dummy window function too.
    """

    def __init__(self, name="DummyPowerSpectrum", min_k=0.02, max_k=0.30, postprocess=None, dummy_window=False, uncert=0.01):
        super().__init__(name=name, postprocess=postprocess, min_k=min_k, max_k=max_k)

        # Set data to camb generated power spectrum
        c = getCambGenerator()
        r_s, pk_lin, _, _ = c.get_data()
        ks = c.ks

        # Apply window function identically as it was applied to data
        # Note this isn't perfect because any error in the window function will propagate
        if dummy_window:
            pk_final = splev(self.w_ks_output, splrep(ks, pk_lin))[self.w_mask]
            self.w_ks_input = self.w_ks_output
            self.w_transform = np.diag(np.ones(self.w_ks_output.size))
            self.w_k0_scale = 0
            self.w_pk = 0
        else:
            pk_sampled = splev(self.w_ks_input, splrep(ks, pk_lin))
            p0 = np.sum(self.w_k0_scale * pk_sampled)
            integral_constraint = self.w_pk * p0
            pk_convolved = np.atleast_2d(pk_sampled) @ self.w_transform
            pk_normalised = (pk_convolved - integral_constraint).flatten()
            pk_final = pk_normalised[self.w_mask]

        # Set covariance to something nice and simple to sample from
        # 1% diagonal uncertainty seems pretty good.
        cov = np.diag((uncert * pk_final) ** 2)
        self.data = pk_final
        self.set_cov(cov)


class DummyCorrelationFunction_SDSS_DR12_Z061_NGC(CorrelationFunction_SDSS_DR12_Z061_NGC):
    """ Dummy correlation function.

    Uses CAMB's linear power spectrum and faked uncertainty.
    """

    def __init__(self, uncert=0.01):
        super().__init__()

        # Set data to camb generated power spectrum
        c = getCambGenerator()
        r_s, pk_lin, _, _ = c.get_data()
        ks = c.ks
        dist = self.data[:, 0]
        xi = pk2xi.PowerToCorrelationGauss(ks).__call__(ks, pk_lin, dist)

        # Set covariance to something nice and simple to sample from
        # 1% diagonal uncertainty seems pretty good.
        cov = np.diag((uncert * xi) ** 2)
        self.data = np.vstack((dist, xi)).T
        self.set_cov(cov)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    dataset = DummyPowerSpectrum_SDSS_DR12_Z061_NGC()
    data = dataset.get_data()

    import matplotlib.pyplot as plt
    import seaborn as sb
    import numpy as np

    plt.errorbar(data["ks"], data["ks"] * data["pk"], yerr=data["ks"] * np.sqrt(np.diag(data["cov"])), fmt="o", c="k")
    plt.show()

    sb.heatmap(data["cov"])
    plt.show()
