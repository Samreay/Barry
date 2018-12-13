import logging
import numpy as np

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.pk2xi import PowerToCorrelationFT, PowerToCorrelationGauss
from barry.framework.cosmology.power_spectrum_smoothing import smooth_hinton2017
from barry.framework.model import Model


class CorrelationPolynomial(Model):

    def __init__(self, fit_omega_m=False, name="BAO Correlation Polynomial Fit"):
        super().__init__(name)

        # Define parameters
        self.fit_omega_m = fit_omega_m
        if self.fit_omega_m:
            self.add_param("om", r"$\Omega_m$", 0.1, 0.5)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("b", r"$b$", 0.01, 10.0)  # Bias
        self.add_param("sigma_nl", r"$\sigma_{NL}$", 1.0, 20.0)  # dampening
        self.add_param("a1", r"$a_1$", -100, 100)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2)  # Polynomial marginalisation 3

        # Set up data structures for model fitting
        self.h0 = 0.6751
        self.camb = CambGenerator(h0=self.h0)

        if not self.fit_omega_m:
            self.omega_m = 0.3121
            self.pk_lin = self.camb.get_data(om=self.omega_m)
        self.pk2xi = PowerToCorrelationGauss(self.camb.ks)
        # self.pk2xi = PowerToCorrelationFT()  # Slower than the Gauss method

        self.nice_data = None  # Place to store things like invert cov matrix

    def get_nice_data(self):
        """ Nice data is a 3-tuple - distance, xi, inv_cov"""
        assert self.data is not None, "data is none. Only invoke after setting data."

        if self.nice_data is None:
            self.nice_data = self.data[0][:, 0], self.data[0][:, 1], np.linalg.inv(self.data[1])
        return self.nice_data

    def compute_correlation_function(self, d, om, alpha, b, sigma_nl, a1, a2, a3):
        """ Computes the correlation function at distance d given the supplied params
        
        Parameters
        ----------
        d : array
            Array of distances in the correlation function to compute
        om : float
            Omega_m
        alpha : float
            Scale applied to distances
        b : float
            Linear bias
        sigma_nl : float
            Dewiggling transition
        a1 : float
            Polynomial shape 1
        a2 : float
            Polynomial shape 2
        a3 : float
            Polynomial shape 3
        
        Returns
        -------
        array
            The correlation function power at the requested distances.
        
        """
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        if self.fit_omega_m:
            pk_lin = self.camb.get_data(om=om, h0=self.h0)
        else:
            pk_lin = self.pk_lin

        # Get the smoothed power spectrum
        pk_smooth = smooth_hinton2017(ks, pk_lin)

        # Blend the two
        pk_linear_weight = np.exp(-0.5 * (ks * sigma_nl)**2)
        pk_dewiggled = pk_linear_weight * pk_lin + (1 - pk_linear_weight) * pk_smooth

        # Conver to correlation function and take alpha into account
        xi = self.pk2xi.pk2xi(ks, pk_dewiggled, d * alpha)

        # Polynomial shape
        shape = a1 / (d ** 2) + a2 / d + a3

        # Add poly shape to xi model, include bias correction
        model = xi * b + shape
        return model

    def get_likelihood(self, *params):
        dist, xi_data, icov = self.get_nice_data()
        if self.fit_omega_m:
            om, alpha, b, sigma_nl, a1, a2, a3 = params
        else:
            alpha, b, sigma_nl, a1, a2, a3 = params
            om = 0.3121
        xi_model = self.compute_correlation_function(dist, om, alpha, b, sigma_nl, a1, a2, a3)

        diff = (xi_data - xi_model)
        chi2 = diff.T @ icov @ diff
        return -0.5 * chi2


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    bao = CorrelationPolynomial(fit_omega_m=True)

    from barry.framework.datasets.mock_correlation import MockAverageCorrelations
    dataset = MockAverageCorrelations()
    data = dataset.get_data()
    bao.set_data(data)

    import timeit
    n = 500

    def test():
        bao.get_likelihood(0.3, 1.0, 1.0, 5.0, 0, 0, 0)
    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if False:
        ss = data[0][:, 0]
        xi = data[0][:, 1]
        xi2 = bao.compute_correlation_function(ss, 0.3, 1, 1, 5, 0, 0, 0)
        xi3 = bao.compute_correlation_function(ss, 0.3, 1, 1, 5, 0, 1, 0)
        xi4 = bao.compute_correlation_function(ss, 0.3, 1, 1, 5, 0, -1, 0)
        import matplotlib.pyplot as plt
        plt.plot(ss, xi, '.', c='b')
        plt.plot(ss, xi2, '.', c='r')
        plt.plot(ss, xi3, '.', c='g')
        plt.plot(ss, xi4, '.', c='y')
        plt.show()
