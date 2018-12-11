import logging
import numpy as np
from scipy.stats import norm

from barry.framework.model import Model


class CorrelationPolynomial(Model):

    def __init__(self):
        super().__init__("BAO Correlation Polynomial Fit")
        self.add_param("om", r"$\Omega_m$", 0.1, 0.6)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("a1", r"$a_1$", -1000, 1000)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -10, 10)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -2, 2)  # Polynomial marginalisation 3


    def get_likelihood(self, omega_m, alpha):
        return np.sum(norm.logpdf(self.data, loc=omega_m, scale=alpha))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    bao = CorrelationPolynomial()
