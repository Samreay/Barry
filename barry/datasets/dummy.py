import logging
from scipy.interpolate import splev, splrep
import numpy as np

from barry.cosmology import pk2xi
from barry.cosmology.camb_generator import getCambGenerator
from barry.datasets import PowerSpectrum_SDSS_DR12, CorrelationFunction_SDSS_DR12_Z061_NGC


class DummyPowerSpectrum_SDSS_DR12(PowerSpectrum_SDSS_DR12):
    """Dummy power spectrum.

    Uses CAMB's linear power spectrum and faked uncertainty. Utilised the SDSS DR12 window function, with option
    to make a dummy window function too.
    """

    def __init__(
        self,
        redshift_bin=3,
        galatic_cap="ngc",
        name="DummyPowerSpectrum",
        min_k=0.02,
        max_k=0.30,
        step_size=1,
        postprocess=None,
        dummy_window=False,
        uncert=0.01,
    ):
        super().__init__(
            redshift_bin=redshift_bin,
            galactic_cap=galatic_cap,
            name=name,
            step_size=step_size,
            postprocess=postprocess,
            min_k=min_k,
            max_k=max_k,
        )

        # Set data to camb generated power spectrum
        c = getCambGenerator()
        pk_lin = c.get_data()["pk_lin"]
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
    """Dummy correlation function.

    Uses CAMB's linear power spectrum and faked uncertainty.
    """

    def __init__(self, name="DummyCorrelationFunction", min_dist=30, max_dist=200, uncert=0.01):
        super().__init__(name=name, min_dist=min_dist, max_dist=max_dist)

        # Set data to camb generated power spectrum
        c = getCambGenerator()
        pk_lin = c.get_data()["pk_lin"]
        ks = c.ks
        dist = self.data[:, 0]
        xi = pk2xi.PowerToCorrelationGauss(ks).__call__(ks, pk_lin, dist)

        # Set covariance to something nice and simple to sample from
        # 1% diagonal uncertainty seems pretty good.
        cov = np.diag((uncert * xi) ** 2)
        self.data = np.vstack((dist, xi)).T
        self.set_cov(cov)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Make plots of the Dummy dataset against the CAMB linear model
    c = getCambGenerator()
    pk_lin = c.get_data()["pk_lin"]

    dataset = DummyPowerSpectrum_SDSS_DR12()
    data = dataset.get_data()
    plt.errorbar(
        data[0]["ks"], data[0]["ks"] * data[0]["pk"], yerr=data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"])), fmt="o", c="k", zorder=1
    )
    plt.errorbar(data[0]["ks"], data[0]["ks"] * splev(dataset.w_ks_output, splrep(c.ks, pk_lin))[dataset.w_mask], fmt="-", c="k", zorder=0)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$k\,P(k)$")
    plt.title(dataset.name)
    plt.show()

    dataset = DummyCorrelationFunction_SDSS_DR12()
    data = dataset.get_data()
    plt.errorbar(
        data[0]["dist"],
        data[0]["dist"] ** 2 * data[0]["xi0"],
        yerr=data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"])),
        fmt="o",
        c="k",
        zorder=1,
    )
    plt.errorbar(
        data[0]["dist"],
        data[0]["dist"] ** 2 * pk2xi.PowerToCorrelationGauss(c.ks).__call__(c.ks, pk_lin, data[0]["dist"]),
        fmt="-",
        c="k",
        zorder=0,
    )
    plt.xlabel(r"$s$")
    plt.ylabel(r"$s^{2}\xi(s)$")
    plt.title(dataset.name)
    plt.show()
