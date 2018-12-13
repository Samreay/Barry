from abc import ABC

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, splev, splrep


class PowerToCorrelation(ABC):
    def pk2xi(self, ks, pk, ss):
        raise NotImplementedError()


class PowerToCorrelationGauss(PowerToCorrelation):
    def __init__(self, ks, interpolateDetail=2, a=0.25):
        super().__init__()
        self.ks = ks
        self.ks2 = np.logspace(np.log(np.min(ks)), np.log(np.max(ks)), interpolateDetail * ks.size, base=np.e)
        self.precomp = self.ks2 * np.exp(-self.ks2 * self.ks2 * a * a) / (2 * np.pi * np.pi) # Precomp a bunch of things

    def pk2xi(self, ks, pks, ss):
        pks2 = interp1d(ks, pks, kind='linear')(self.ks2)
        # Set up output array
        xis = np.zeros(ss.size)

        # Precompute k^2 and gauss (note missing a ks factor below because integrating in log space)
        kkpks = self.precomp * pks2

        # Iterate over all values in desired output array of distances (s)
        for i, s in enumerate(ss):
            integrand = kkpks * np.sin(self.ks2 * s) / s
            xis[i] = trapz(integrand, self.ks2)

        return xis


class PowerToCorrelationFT(PowerToCorrelation):
    def __init__(self):
        from hankel import SymmetricFourierTransform
        self.ft = SymmetricFourierTransform(ndim=3, N=2000, h=0.001)

    def pk2xi(self, ks, pk, ss):
        pkspline = splrep(ks, pk)
        f = lambda k: splev(k, pkspline)
        xi = self.ft.transform(f, ss, inverse=True, ret_err=False)
        return xi


if __name__ == "__main__":
    from barry.framework.cosmology.camb_generator import CambGenerator
    camb = CambGenerator(h0=70.0)
    ks = camb.ks
    pklin, pknl = camb.get_data(0.3, 0.70)

    ss = np.linspace(30, 200, 85)

    pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1)
    pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25)
    pk2xi_ft = PowerToCorrelationFT()

    if True:
        import timeit
        n = 200

        def test():
            pk2xi_gauss.pk2xi(ks, pklin, ss)
        print("Gauss method: %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

        def test2():
            pk2xi_ft.pk2xi(ks, pklin, ss)
        print("FT method: %.2f milliseconds" % (timeit.timeit(test2, number=n) * 1000 / n))

    if True:
        import matplotlib.pyplot as plt
        xi1 = pk2xi_gauss.pk2xi(ks, pklin, ss)
        xi2 = pk2xi_ft.pk2xi(ks, pklin, ss)
        xi_good = pk2xi_good.pk2xi(ks, pklin, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, xi_good, '.', c='k')
        ax[0].plot(ss, xi1, '.', c='b', label="Gauss")
        ax[0].plot(ss, xi2, '.', c='r', label="FT")
        ax[0].legend()
        ax[1].plot(ss, 100000 * (xi_good - xi1), '.', c='b')
        ax[1].plot(ss, 100000 * (xi_good - xi2), '.', c='r')
        ax[1].axhline(0)
        ax[1].set_xlabel("Dist")
        ax[1].set_ylabel("100 000 times diff")
        ax[0].set_ylabel("xi(s)")
        plt.show()

