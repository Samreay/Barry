import logging

from abc import ABC

import sys
import numpy as np
import numpy.fft as fft
from scipy.special import spherical_jn, gamma, loggamma
from scipy.integrate import trapz
from scipy.interpolate import interp1d, splev, splrep
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative


class PowerToCorrelation(ABC):
    """Generic class for converting power spectra to correlation functions

    Using a class based method as there might be multiple implementations and
    some of the implementations have state.
    """

    def __init__(self, ell=0):
        self.ell = ell

    def __call__(self, ks, pk, ss):
        """Generates the correlation function

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
    """A pk2xi implementation using manual numeric integration with Gaussian dampening factor"""

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
                bessel = (1.0 - 3.0 / z**2) * np.sin(z) / s + 3.0 * np.cos(z) / (z * s)
            elif self.ell == 4:
                bessel = (105.0 / z**4 - 45.0 / z**2 + 1.0) * np.sin(z) / s - (105.0 / z**2 - 10.0) * np.cos(z) / (z * s)
            else:
                bessel = spherical_jn(self.ell, z)
            integrand = kkpks * bessel
            xis[i] = trapz(integrand, self.ks2)

        return xis


class PowerToCorrelationFT(PowerToCorrelation):
    """A pk2xi implementation utilising the Hankel library to use explicit FFT."""

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
        f = lambda k: splev(k, pkspline) / (k**self.ell)
        xi = (2.0 * np.pi * ss) ** self.ell * (1j) ** self.ell * self.ft.transform(f, ss, inverse=True, ret_err=False)
        return xi


class PowerToCorrelationFFTLog(PowerToCorrelation):
    """A pk2xi implementation based on Ashley Ross' code."""

    def __init__(self, ell=0, q=1.5, r0=10.0, transformed_axis=0, output_r_power=-3, n=None):
        """

        Parameters
        ----------
        num_nodes : int, optional
            Number of nodes in FFT
        h : float, optional
            Step size of integration
        """

        super().__init__(ell=ell)
        self.q = q
        self.r0 = r0
        self.transformed_axis = transformed_axis
        self.output_r_power = output_r_power
        self.n = self.q if n is None else n
        self.mu = ell + 0.5

    def __call__(self, ks, pk, ss):
        """
        Hankel transform, with power law biasing
        a'(r) = r**(output_r_power) integral_0^infty a(k) (kr)^q J_mu(kr) r dk
        with k logathmically spaced
        based on Hamilton 2000 FFTLog algorithm.

        Args:
             k : 1D numpy array, log spacing defining the rectangular k-grid of Pk
             a : 1D or 2D numpy array, axis to be transformed must have same size as k
             q : float or int, power of kr in transformation
             mu : float or int, parameter of Bessel function
        Options:
             r0 : float, default is 10 (units of 1/k)
             transformed_axis : int, if a is 2D, can transform along axis 0 or 1,
                                do the transform on all rows of other axis.
             ss : 1D numpy array or None. if set, output is interpolated to
                                this array of coordinates
             output_r_power : multiply output by r**output_r_power
             n : default n=None and set to n=q , do not play with this
        Returns:
             r : 1D numpy array
             a'(r) : numpy array, 1D or 2D, depending on input a
        """
        if len(pk.shape) > 2:
            print("not implemented for a of more than 2D")
            sys.exit(12)

        k0 = ks[0]
        N = len(ks)
        L = np.log(ks.max() / k0) * N / (N - 1.0)  ## this is important, need to have the right scale !!
        emm = N * np.fft.fftfreq(N)

        nout = self.n + self.output_r_power

        x = (self.q - self.n) + 2 * np.pi * 1j * emm / L  # Eq. 174

        if 1:  # choose r0 to limit ringing with the condition u(-N/2)=u(N/2), see Hamilton 2000, Eq. 186
            x0 = (self.q - self.n) + np.pi * 1j * N / L
            tmp = 1.0 / np.pi * np.angle(2**x0 * gamma((self.mu + 1 + x0) / 2.0) / gamma((self.mu + 1 - x0) / 2.0))
            number = int(np.log(k0 * self.r0) * N / L - tmp)
            lowringing_r0 = np.exp(L / N * (tmp + number)) / k0
            r0 = lowringing_r0

        um = (
            (k0 * r0) ** (-2 * np.pi * 1j * emm / L) * 2**x * (gamma((self.mu + 1 + x) / 2.0) / gamma((self.mu + 1 - x) / 2.0))
        )  # Eq. 174
        um[0] = um[0].real

        r = r0 * np.exp(-emm * L / N)
        s = np.argsort(r)
        rs = r[s]  # sorted

        if len(pk.shape) == 1:
            transformed = (fft.ifft(um * fft.fft(pk * (ks**self.n))) * (r**nout)).real[s]
            if ss is not None:
                transformed = self.extrap(ss, rs, transformed)
        else:
            if self.transformed_axis == 0:
                if ss is None:
                    transformed = np.zeros_like(pk)
                    for i in xrange(pk.shape[1]):  # I don't know how to do this at once
                        transformed[:, i] = (fft.ifft(um * fft.fft(pk[:, i] * (ks**self.n))) * (r**nout)).real[s]
                else:
                    transformed = np.zeros(shape=(ss.size, pk.shape[1]), dtype=pk.dtype)
                    for i in xrange(pk.shape[1]):
                        transformed[:, i] = self.extrap(ss, rs, (fft.ifft(um * fft.fft(pk[:, i] * (ks**self.n))) * (r**nout)).real[s])
            else:
                if ss is None:
                    transformed = np.zeros_like(pk)
                    for i in xrange(pk.shape[0]):
                        transformed[i, :] = (fft.ifft(um * fft.fft(pk[i, :] * (ks**self.n))) * (r**nout)).real[s]
                else:
                    transformed = np.zeros(shape=(pk.shape[0], ss.size), dtype=pk.dtype)
                    for i in xrange(pk.shape[0]):
                        transformed[i, :] = self.extrap(ss, rs, (fft.ifft(um * fft.fft(pk[i, :] * (ks**self.n))) * (r**nout)).real[s])

        return np.real((1j) ** self.ell * transformed / (5.0 * np.pi))

    def extrap(self, x, xp, yp):
        """np.interp function with linear extrapolation"""
        y = np.interp(x, xp, yp)
        y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
        y = np.where(x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y)
        return y

    def extrapolate_pk_logspace(self, ko, ki, pki):
        """
        Linear interpolation of pk in log,log,
        taking care of extrapolation (also linear in log,log)
        and numerical precision
        """
        logpk = self.extrap(np.log(ko), np.log(ki[pki > 0]), np.log(pki[pki > 0]))
        pk = np.zeros(logpk.shape)
        minlogpk = -np.log(np.finfo(np.float64).max)
        pk[logpk > minlogpk] = np.exp(logpk[logpk > minlogpk])
        return pk


class PowerToCorrelationSphericalBessel(PowerToCorrelation):
    def __init__(self, qs=None, ell=15, low_ring=True, fourier=True):

        """
        From Stephen Chen. Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.

        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.

        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.

        Taken from velocileptors.

        """

        if qs is None:
            qs = np.logspace(-4, np.log(5.0), 2000)

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2 * np.pi**2)

        self.q = qs
        self.ell = ell

        self.Nx = len(qs)
        self.Delta = np.log(qs[-1] / qs[0]) / (self.Nx - 1)

        self.N = 2 ** (int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.pads = np.zeros((self.N - self.Nx) // 2)
        self.pad_iis = np.arange(self.Npad - self.Npad // 2, self.N - self.Npad // 2)

        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N // 2 + 1)
        self.ydict = {}
        self.udict = {}
        self.qdict = {}

        if low_ring:
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
                ys = np.exp(lnxy - self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

    def __call__(self, ks, fq, ss, damping=0.25, nu=None):
        """
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        """
        if nu is None:
            nu = self.ell

        fq = fq * np.exp(-(ks**2) * damping**2)

        q = self.qdict[nu]
        y = self.ydict[nu]
        f = np.concatenate((self.pads, self.q ** (3 - q) * fq, self.pads))

        fks = np.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = np.fft.hfft(gks) / self.N

        return np.real((1j) ** nu * splev(ss, splrep(y, y ** (-q) * gs[self.pad_iis])))

    def UK(self, nu, z):
        """
        The Mellin transform of the spherical bessel transform.
        """
        return self.sqrtpi * np.exp(np.log(2) * (z - 2) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (3 + nu - z)))

    def update_tilt(self, nu, tilt):
        """
        Update the tilt for a particular nu. Assume low ring coordinates.
        """
        q = tilt
        ll = nu

        ms = np.arange(0, self.N // 2 + 1)
        lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
        ys = np.exp(lnxy - self.Delta) * self.q / (self.q[0] * self.q[-1])
        us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

        self.ydict[ll] = ys
        self.udict[ll] = us
        self.qdict[ll] = q

    def loginterp(
        x,
        y,
        yint=None,
        side="both",
        lorder=9,
        rorder=9,
        lp=1,
        rp=-2,
        ldx=1e-6,
        rdx=1e-6,
        interp_min=-12,
        interp_max=12,
        Nint=10**5,
        verbose=False,
        option="B",
    ):
        """
        Extrapolate function by evaluating a log-index of left & right side.

        From Chirag Modi's CLEFT code at
        https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py

        The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
        """

        if yint is None:
            yint = interpolate(x, y, k=5)
        if side == "both":
            side = "lr"

        # Make sure there is no zero crossing between the edge points
        # If so assume there can't be another crossing nearby

        if np.sign(y[lp]) == np.sign(y[lp - 1]) and np.sign(y[lp]) == np.sign(y[lp + 1]):
            l = lp
        else:
            l = lp + 2

        if np.sign(y[rp]) == np.sign(y[rp - 1]) and np.sign(y[rp]) == np.sign(y[rp + 1]):
            r = rp
        else:
            r = rp - 2

        lneff = derivative(yint, x[l], dx=x[l] * ldx, order=lorder) * x[l] / y[l]
        rneff = derivative(yint, x[r], dx=x[r] * rdx, order=rorder) * x[r] / y[r]

        # print(lneff, rneff)

        # uncomment if you like warnings.
        # if verbose:
        #    if lneff < 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
        #    if rneff > 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

        if option == "A":

            xl = np.logspace(interp_min, np.log10(x[l]), Nint)
            xr = np.logspace(np.log10(x[r]), interp_max, Nint)
            yl = y[l] * (xl / x[l]) ** lneff
            yr = y[r] * (xr / x[r]) ** rneff
            # print(xr/x[r])

            xint = x[l + 1 : r].copy()
            yint = y[l + 1 : r].copy()
            if side.find("l") > -1:
                xint = np.concatenate((xl, xint))
                yint = np.concatenate((yl, yint))
            if side.find("r") > -1:
                xint = np.concatenate((xint, xr))
                yint = np.concatenate((yint, yr))
            yint2 = interpolate(xint, yint, k=5, ext=3)

        else:
            # nan_to_numb is to prevent (xx/x[l/r])^lneff to go to nan on the other side
            # since this value should be zero on the wrong side anyway
            # yint2 = lambda xx: (xx <= x[l]) * y[l]*(xx/x[l])**lneff \
            #                 + (xx >= x[r]) * y[r]*(xx/x[r])**rneff \
            #                 + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)
            yint2 = (
                lambda xx: (xx <= x[l]) * y[l] * np.nan_to_num((xx / x[l]) ** lneff)
                + (xx >= x[r]) * y[r] * np.nan_to_num((xx / x[r]) ** rneff)
                + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k=5, ext=3)(xx)
            )

        return yint2


if __name__ == "__main__":

    sys.path.append("../..")
    import timeit
    import matplotlib.pyplot as plt
    from barry.cosmology.camb_generator import getCambGenerator, Omega_m_z

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    om = 0.31
    c = getCambGenerator()
    ks = c.ks
    pklin = c.get_data(om=om)["pk_lin_z"]
    growth = Omega_m_z(0.31, c.redshift) ** 0.55

    ss = np.linspace(30, 200, 85)

    # Compare Gaussian with many narrow bins, fewer bins, and Hankel transform
    pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1)
    pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25)
    pk2xi_ft = PowerToCorrelationFT()
    pk2xi_fftlog = PowerToCorrelationFFTLog()
    pk2xi_sphericalbessel = PowerToCorrelationSphericalBessel(qs=ks, ell=4, fourier=True, low_ring=False)

    if True:
        n = 200

        def test_good():
            pk2xi_good.__call__(ks, pklin, ss)

        def test_gauss():
            pk2xi_gauss.__call__(ks, pklin, ss)

        def test_ft():
            pk2xi_ft.__call__(ks, pklin, ss)

        def test_fftlog():
            pk2xi_fftlog.__call__(ks, pklin, ss)

        def test_sphericalbessel():
            pk2xi_sphericalbessel.__call__(ks, pklin, ss)

        print("Gauss-Narrow method: %.2f milliseconds" % (timeit.timeit(test_good, number=n) * 1000 / n))

        print("Gauss method: %.2f milliseconds" % (timeit.timeit(test_gauss, number=n) * 1000 / n))

        print("FT method: %.2f milliseconds" % (timeit.timeit(test_ft, number=n) * 1000 / n))

        print("FFTLog method: %.2f milliseconds" % (timeit.timeit(test_fftlog, number=n) * 1000 / n))

        print("Spherical Bessel method: %.2f milliseconds" % (timeit.timeit(test_sphericalbessel, number=n) * 1000 / n))

    if True:
        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=0.1, ell=0)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=0)
        pk2xi_ft = PowerToCorrelationFT(ell=0)
        pk2xi_fftlog = PowerToCorrelationFFTLog(ell=0)
        pk2xi_sphericalbessel = PowerToCorrelationSphericalBessel(qs=ks, ell=0, fourier=True, low_ring=False)

        pk = (1.0 + 2.0 / 3.0 * growth + 1.0 / 5.0 * growth**2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)
        xi_fftlog = pk2xi_fftlog.__call__(ks, pk, ss)
        xi_sphericalbessel = pk2xi_sphericalbessel.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss**2 * xi_good, ".", c="k")
        # ax[0].plot(ss, ss**2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss**2 * xi2, ".", c="r", label="FT")
        ax[0].plot(ss, ss**2 * xi_fftlog, ".", c="g", label="FFTLog")
        ax[0].plot(ss, ss**2 * xi_sphericalbessel, ".", c="purple", label="SphericalBessel")
        ax[0].legend()
        # ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].plot(ss, 100.0 * (xi_good - xi_fftlog), ".", c="g")
        ax[1].plot(ss, 100.0 * (xi_good - xi_sphericalbessel), ".", c="purple")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{0}(s)$")
        plt.title(r"$\ell=0$")
        plt.show()

        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1, ell=2)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=2)
        pk2xi_ft = PowerToCorrelationFT(ell=2)
        pk2xi_fftlog = PowerToCorrelationFFTLog(ell=2)
        pk2xi_sphericalbessel = PowerToCorrelationSphericalBessel(qs=ks, ell=2, fourier=True, low_ring=False)

        pk = (4.0 / 3.0 * growth + 4.0 / 7.0 * growth**2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)
        xi_fftlog = pk2xi_fftlog.__call__(ks, pk, ss)
        xi_sphericalbessel = pk2xi_sphericalbessel.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss**2 * xi_good, ".", c="k")
        # ax[0].plot(ss, ss**2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss**2 * xi2, ".", c="r", label="FT")
        ax[0].plot(ss, ss**2 * xi_fftlog, ".", c="g", label="FFTLog")
        ax[0].plot(ss, ss**2 * xi_sphericalbessel, ".", c="purple", label="SphericalBessel")
        ax[0].legend()
        # ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].plot(ss, 100.0 * (xi_good - xi_fftlog), ".", c="g")
        ax[1].plot(ss, 100.0 * (xi_good - xi_sphericalbessel), ".", c="purple")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{2}(s)$")
        plt.title(r"$\ell=2$")
        plt.show()

        pk2xi_good = PowerToCorrelationGauss(ks, interpolateDetail=10, a=1, ell=4)
        pk2xi_gauss = PowerToCorrelationGauss(ks, interpolateDetail=2, a=0.25, ell=4)
        pk2xi_ft = PowerToCorrelationFT(ell=4)
        pk2xi_fftlog = PowerToCorrelationFFTLog(ell=4)
        pk2xi_sphericalbessel = PowerToCorrelationSphericalBessel(qs=ks, ell=4, fourier=True, low_ring=False)

        pk = (8.0 / 35.0 * growth**2) * pklin
        xi1 = pk2xi_gauss.__call__(ks, pk, ss)
        xi2 = pk2xi_ft.__call__(ks, pk, ss)
        xi_good = pk2xi_good.__call__(ks, pk, ss)
        xi_fftlog = pk2xi_fftlog.__call__(ks, pk, ss)
        xi_sphericalbessel = pk2xi_sphericalbessel.__call__(ks, pk, ss)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(ss, ss**2 * xi_good, ".", c="k")
        # ax[0].plot(ss, ss**2 * xi1, ".", c="b", label="Gauss")
        ax[0].plot(ss, ss**2 * xi2, ".", c="r", label="FT")
        ax[0].plot(ss, ss**2 * xi_fftlog, ".", c="g", label="FFTLog")
        ax[0].plot(ss, ss**2 * xi_sphericalbessel, ".", c="purple", label="SphericalBessel")
        ax[0].legend()
        # ax[1].plot(ss, 100.0 * (xi_good - xi1), ".", c="b")
        ax[1].plot(ss, 100.0 * (xi_good - xi2), ".", c="r")
        ax[1].plot(ss, 100.0 * (xi_good - xi_fftlog), ".", c="g")
        ax[1].plot(ss, 100.0 * (xi_good - xi_sphericalbessel), ".", c="purple")
        ax[1].axhline(0)
        ax[1].set_xlabel(r"$s$")
        ax[1].set_ylabel(r"$100 \times \mathrm{diff}$")
        ax[0].set_ylabel(r"$\xi_{4}(s)$")
        plt.title(r"$\ell=4$")
        plt.show()
