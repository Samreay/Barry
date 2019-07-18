import logging
from functools import lru_cache

import numpy as np

import sys
sys.path.append("../../..")

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.pk2xi import PowerToCorrelationFT, PowerToCorrelationGauss
from barry.framework.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth
from barry.framework.model import Model


class CorrelationPolynomial(Model):

    def __init__(self, smooth_type="hinton2017", name="BAO Correlation Polynomial Fit", fix_params=['om'], smooth=False):
        super().__init__(name)

        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.camb = CambGenerator()
        self.h0 = self.camb.h0
        self.smooth = smooth
        self.pk2xi = PowerToCorrelationGauss(self.camb.ks)

    def declare_parameters(self):
        # Define parameters
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.3121)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch
        self.add_param("sigma_nl", r"$\Sigma_{NL}$", 1.0, 20.0, 1.0)  # dampening
        self.add_param("b", r"$b$", 0.01, 10.0, 1.0)  # Bias
        self.add_param("a1", r"$a_1$", -100, 100, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2, 0)  # Polynomial marginalisation 3

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed, linear power spectrum and the wiggle ratio

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        r_s, pk_lin = self.camb.get_data(om=om, h0=self.h0)
        pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=om, h0=self.h0)  # Get the smoothed power spectrum
        return pk_lin, pk_smooth_lin

    def compute_correlation_function(self, d, p, smooth=False):
        """ Computes the correlation function at distance d given the supplied params
        
        Parameters
        ----------
        d : array
            Array of distances in the correlation function to compute
        params : dict
            dictionary of parameter name to float value pairs

        Returns
        -------
        array
            The correlation function power at the requested distances.
        
        """
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        pk_lin, pk_smooth = self.compute_basic_power_spectrum(p["om"])

        # Blend the two
        pk_linear_weight = np.exp(-0.5 * (ks * p["sigma_nl"])**2)
        pk_dewiggled = pk_linear_weight * pk_lin + (1 - pk_linear_weight) * pk_smooth

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi.pk2xi(ks, pk_dewiggled, d * p["alpha"])

        # Polynomial shape
        shape = p["a1"] / (d ** 2) + p["a2"] / d + p["a3"]

        # Add poly shape to xi model, include bias correction
        model = xi * p["b"] + shape
        return model

    def get_model(self, p, smooth=False):
        pk_model = self.compute_correlation_function(self.data["dist"], p, smooth=smooth)
        return pk_model

    def get_likelihood(self, p):
        d = self.data
        xi_model = self.get_model(p, smooth=smooth)

        diff = (d["xi"] - xi_model)
        chi2 = diff.T @ d["icov"] @ diff
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
        bao.get_likelihood(0.3, 1.0, 5.0, 1.0, 0, 0, 0)
    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ss = data["dist"]
        xi = data["xi"]
        xi2 = bao.compute_correlation_function(ss, 0.3, 1, 5, 1, 0, 0, 0)
        xi3 = bao.compute_correlation_function(ss, 0.3, 1, 5, 1, 0, 1, 0)
        bao.smooth_type="eh1998"
        xi4 = bao.compute_correlation_function(ss, 0.3, 1, 5, 1, 0, 1, 0)
        import matplotlib.pyplot as plt
        plt.errorbar(ss, xi, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
        plt.plot(ss, xi2, '.', c='r')
        plt.plot(ss, xi3, '.', c='g')
        plt.plot(ss, xi4, '.', c='y')
        plt.show()
