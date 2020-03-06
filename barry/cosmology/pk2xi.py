import logging

from abc import ABC

import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import trapz
from scipy.interpolate import interp1d, splev, splrep


class PowerToCorrelation(ABC):
    """ Generic class for converting power spectra to correlation functions

    Using a class based method as there might be multiple implementations and
    some of the implementations have state.
    """

    def __init__(self, ell=0):
        self.ell = ell

    def __call__(self, ks, pk, ss):
        """ Generates the correlation function

        Parameters
        ----------
        ks : np.ndarray
            The k values for the power spectrum data. *Assumed to be in log space*
        pk : np.ndarray
            The P(k) values
        ss : np.nparray
            The distances to calculate xi(s) at.

        Returns
        -------
        xi : np.ndarray
            The correlation function at the specified distances
        """
        raise NotImplementedError()


class PowerToCorrelationGauss(PowerToCorrelation):
    """ A pk2xi implementation using manual numeric integration with Gaussian dampening factor
    """

    def __init__(self, ks, interpolateDetail=2, a=0.25, ell=0):
        super().__init__(ell=ell)
        self.ks = ks
        self.ks2 = np.logspace(np.log(np.min(ks)), np.log(np.max(ks)), interpolateDetail * ks.size, base=np.e)
        self.precomp = self.ks2 * np.exp(-self.ks2 * self.ks2 * a * a) / (2 * np.pi * np.pi)  # Precomp a bunch of things

    def __call__(self, ks, pks, ss):
        pks2 = interp1d(ks, pks, kind="linear")(self.ks2)
        # Set up output array
        xis = np.zeros(ss.size)

        # Precompute k^2 and gauss (note missing a ks factor below because integrating in log space)
        kkpks = self.precomp * pks2

        # Iterate over all values in desired output array of distances (s)
        for i, s in enumerate(ss):
            z = self.ks2 * s
            if self.ell == 0:
                bessel = np.sin(z) / s
            elif self.ell == 2:
                bessel = (1.0 - 3.0 / z ** 2) * np.sin(z) / s + 3.0 * np.cos(z) / (z * s)
            elif self.ell == 4:
                bessel = (105.0 / z ** 4 - 45.0 / z ** 2 + 1.0) * np.sin(z) / s - (105.0 / z ** 2 - 10.0) * np.cos(z) / (z * s)
            else:
                bessel = spherical_jn(self.ell, z)
            integrand = kkpks * bessel
            xis[i] = trapz(integrand, self.ks2)

        return xis


class PowerToCorrelationFT(PowerToCorrelation):
    """ A pk2xi implementation utilising the Hankel library to use explicit FFT.
    """

    def __init__(self, num_nodes=None, h=0.001, ell=0):
        """

        Parameters
        ----------
         num_nodes : int, optional
            Number of nodes in FFT
        h : float, optional
            Step size of integration
        """
        from hankel import SymmetricFourierTransform

        super().__init__(ell=ell)
        self.ft = SymmetricFourierTransform(ndim=2 * ell + 3, N=num_nodes, h=h)

    def __call__(self, ks, pk, ss):
        pkspline = splrep(ks, pk)
        f = lambda k: splev(k, pkspline) / (k ** self.ell)
        xi = (2.0 * np.pi * ss) ** self.ell * self.ft.transform(f, ss, inverse=True, ret_err=False)
        return xi


if __name__ == "__main__":

    import timeit
    import matplotlib.pyplot as plt
    from barry.cosmology.camb_generator import getCambGenerator, Omega_m_z

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    om = 0.31
    c = getCambGenerator()
    ks = c.ks
    pklin = c.get_data(om=om)["pk_lin"]
    growth = Omega_m_z(0.31, c.redshift) ** 0.55

    ss = np.linspace(30, 200, 85)

    # Compare Gaussian with many narrow bins, fewer bins, and Hankel transform
    pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1)
    pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25)
    pk2xi_ft = PowerToCorrelationFT()

    if True:
        n = 200

        def test_good():
            pk2xi_good.__call__(ks, pklin, ss)

        def test_gauss():
            pk2xi_gauss.__call__(ks, pklin, ss)

        def test_ft():
            pk2xi_ft.__call__(ks, pklin, ss)

        print("Gauss-Narrow method: %.2f milliseconds" % (timeit.timeit(test_good, number=n) * 1000 / n))

        print("Gauss method: %.2f milliseconds" % (timeit.timeit(test_gauss, number=n) * 1000 / n))

        print("FT method: %.2f milliseconds" % (timeit.timeit(test_ft, number=n) * 1000 / n))

    if True:
        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1, ell=0)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=0)
        pk2xi_ft = PowerToCorrelationFT(ell=0)

        pk = (1.0 + 2.0 / 3.0 * growth + 1.0 / 5.0 * growth ** 2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss ** 2 * xi_good, ".", c="k")
        ax[0].plot(ss, ss ** 2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss ** 2 * xi2, ".", c="r", label="FT")
        ax[0].legend()
        ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{0}(s)$")
        plt.title(r"$\ell=0$")
        plt.show()

        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1, ell=2)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=2)
        pk2xi_ft = PowerToCorrelationFT(ell=2)

        pk = (4.0 / 3.0 * growth + 4.0 / 7.0 * growth ** 2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss ** 2 * xi_good, ".", c="k")
        ax[0].plot(ss, ss ** 2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss ** 2 * xi2, ".", c="r", label="FT")
        ax[0].legend()
        ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{2}(s)$")
        plt.title(r"$\ell=2$")
        plt.show()

        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1, ell=4)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=4)
        pk2xi_ft = PowerToCorrelationFT(ell=4)

        pk = (8.0 / 35.0 * growth ** 2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss ** 2 * xi_good, ".", c="k")
        ax[0].plot(ss, ss ** 2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss ** 2 * xi2, ".", c="r", label="FT")
        ax[0].legend()
        ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{4}(s)$")
        plt.title(r"$\ell=4$")
        plt.show()

    # Test the impact of cutting the power spectrum at lower k_max.
    if True:
        thresh = 1.0
        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1.0)
        pk2xi_cut = PowerToCorrelationGauss(ks[ks < thresh], interpolateDetail=10, a=1.0)
        xi1 = pk2xi_cut.__call__(ks[ks < thresh], pklin[ks < thresh], ss)
        xi_good = pk2xi_good.__call__(ks, pklin, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, xi_good, ".", c="k")
        ax[0].plot(ss, xi1, ".", c="b", label="k < 2.0")
        ax[0].legend()
        ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].axhline(0)
        ax[1].set_xlabel("s")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi(s)$")
        plt.show()
