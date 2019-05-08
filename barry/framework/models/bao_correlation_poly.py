import logging
import numpy as np

import sys
sys.path.append("../../..")

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.pk2xi import PowerToCorrelationFT, PowerToCorrelationGauss
from barry.framework.cosmology.power_spectrum_smoothing import smooth_hinton2017, smooth_eh1998
from barry.framework.model import Model


class CorrelationPolynomial(Model):

    def __init__(self, fit_omega_m=False, smooth_type="hinton2017", name="BAO Correlation Polynomial Fit"):
        super().__init__(name)

        self.smooth_type=smooth_type
        if ((smooth_type != "hinton2017") and (smooth_type != "eh1998")):
            print("smooth_type not recognised, must be either: 'hinton2017' (default) or 'eh1998'.")
            exit(0)

        # Define parameters
        self.fit_omega_m = fit_omega_m
        if self.fit_omega_m:
            self.add_param("om", r"$\Omega_m$", 0.1, 0.5)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("sigma_nl", r"$\Sigma_{NL}$", 1.0, 20.0)  # dampening
        self.add_param("b", r"$b$", 0.01, 10.0)  # Bias
        self.add_param("a1", r"$a_1$", -100, 100)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2)  # Polynomial marginalisation 3

        # Set up data structures for model fitting
        self.h0 = 0.6751
        self.camb = CambGenerator(h0=self.h0)

        if not self.fit_omega_m:
            self.omega_m = 0.3121
            self.r_s, self.pk_lin = self.camb.get_data(om=self.omega_m)
        self.pk2xi = PowerToCorrelationGauss(self.camb.ks)
        # self.pk2xi = PowerToCorrelationFT()  # Slower than the Gauss method

        self.nice_data = None  # Place to store things like invert cov matrix

    def compute_correlation_function(self, d, om, alpha, sigma_nl, b, a1, a2, a3):
        """ Computes the correlation function at distance d given the supplied params
        
        Parameters
        ----------
        d : array
            Array of distances in the correlation function to compute
        om : float
            Omega_m
        alpha : float
            Scale applied to distances
        sigma_nl : float
            Dewiggling transition
        b : float
            Linear bias
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
            r_s, pk_lin = self.camb.get_data(om=om, h0=self.h0)
        else:
            pk_lin = self.pk_lin

        # Get the smoothed power spectrum
        if (self.smooth_type == "hinton2017"):
            pk_smooth = smooth_hinton2017(ks, pk_lin)
        elif (self.smooth_type == "eh1998"):
            pk_smooth = smooth_eh1998(ks, pk_lin, om=om, h0=self.h0)
        else:
            print("self.smooth_type not recognised, must be either: 'hinton2017' (default) or 'eh1998'.")
            exit(0)

        # Blend the two
        pk_linear_weight = np.exp(-0.5 * (ks * sigma_nl)**2)
        pk_dewiggled = pk_linear_weight * pk_lin + (1 - pk_linear_weight) * pk_smooth

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi.pk2xi(ks, pk_dewiggled, d * alpha)

        # Polynomial shape
        shape = a1 / (d ** 2) + a2 / d + a3

        # Add poly shape to xi model, include bias correction
        model = xi * b + shape
        return model

    def get_likelihood(self, *params):
        d = self.data
        if self.fit_omega_m:
            om, alpha, sigma_nl, b, a1, a2, a3 = params
        else:
            alpha, sigma_nl, b, a1, a2, a3 = params
            om = 0.3121
        xi_model = self.compute_correlation_function(d["dist"], om, alpha, sigma_nl, b, a1, a2, a3)

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
