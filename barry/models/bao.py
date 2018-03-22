from barry.model import Model
import numpy as np


class BAOModel(Model):

    def __init__(self):
        super().__init__("BAO")

    def get_prior(self, data, params):
        omega_m, alpha = params
        if not(0 < omega_m < 1) or not(0.8 < alpha < 1.2):
            return -np.inf
        return 0

    def get_likelihood(self, data, params):
        omega_m, alpha = params
        from scipy.stats import norm
        return norm.logpdf(omega_m, 0.3, 0.1) + norm.logpdf(alpha, 1.0, 0.02)


if __name__ == "__main__":
    bao = BAOModel()

