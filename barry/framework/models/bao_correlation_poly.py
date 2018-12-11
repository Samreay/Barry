import logging
import numpy as np

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.pk2xi import pk2xiGauss
from barry.framework.cosmology.power_spectrum_smoothing import smooth_hinton2017
from barry.framework.model import Model


class CorrelationPolynomial(Model):

    def __init__(self):
        super().__init__("BAO Correlation Polynomial Fit")

        # Define parameters
        self.add_param("om", r"$\Omega_m$", 0.1, 0.6)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("b", r"$b$", 0.01, 12.0)  # Bias
        self.add_param("sigma_nl", r"$\sigma_{NL}$", 2.0, 20.0)  # dampening
        self.add_param("a1", r"$a_1$", -1000, 1000)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -10, 10)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -2, 2)  # Polynomial marginalisation 3

        # Set up data structures for model fitting
        self.h0 = 70.0
        self.camb = CambGenerator(h0=self.h0)

        self.nice_data = None # Place to store things like invert cov matrix

    def get_nice_data(self):
        """ Nice data is a 3-tuple - distance, xi, inv_cov"""
        assert self.data is not None, "data is none. Only invoke after setting data."

        if self.nice_data is None:
            self.nice_data = self.data[0][:, 0], self.data[0][:, 1], np.linalg.inv(self.data[1])
        return self.nice_data

    def compute_correlation_function(self, distances, om, alpha, b, sigma_nl, a1, a2, a3):
        omch2 = om * (self.h0 / 100)**2

        # Get base linear power spectrum from camb
        ks = self.camb.ks
        pk_lin, pk_nl = self.camb.get_data(omch2=omch2, h0=self.h0)
        # TODO: Figure out if I should be using the linear or non-linear model here

        # Get the smoothed power spectrum
        pk_smooth = smooth_hinton2017(ks, pk_nl)

        # Blend the two
        pk_linear_weight = np.exp(-0.5 * (ks * sigma_nl)**2)
        pk_dewiggled = pk_linear_weight * pk_lin + (1 - pk_linear_weight) * pk_smooth

        # Conver to correlation function and take alpha into account
        ss = distances / alpha
        xi = pk2xiGauss(ks, pk_dewiggled, ss)

        # Polynomial shape
        d = distances
        shape = a1 / (d ** 2) + a2 / d + a3

        # Add poly shape to xi model, include bias correction
        model = xi * b + shape
        return model

    def get_likelihood(self, om, alpha, b, sigma_nl, a1, a2, a3):
        dist, xi_data, icov = self.get_nice_data()
        xi_model = self.compute_correlation_function(dist, om, alpha, b, sigma_nl, a1, a2, a3)

        diff = (xi_data - xi_model)
        chi2 = diff.T @ icov @ diff
        return -0.5 * chi2


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    bao = CorrelationPolynomial()

    from barry.framework.datasets.mock_correlation import MockAverageCorrelations
    dataset = MockAverageCorrelations()
    data = dataset.get_data()
    bao.set_data(data)

    # print(bao.get_likelihood(0.5, 1.1, 1.0, 5.0, 0, 0, 0))
    import timeit
    n = 100

    def test():
        bao.get_likelihood(0.3, 1.0, 1.0, 5.0, 0, 0, 0)
    #print("Takes on average, %.3f seconds" % (timeit.timeit(test, number=n) / n))

    ss = data[0][:, 0]
    xi = data[0][:, 1]
    xi2 = bao.compute_correlation_function(ss, 0.3, 1, 1, 5, 0, 0, 0)
    import matplotlib.pyplot as plt
    plt.plot(ss, ss*ss*xi, '.', c='b')
    plt.plot(ss, ss*ss*xi2, '.', c='r')
    plt.show()
