import numpy as np
from barry.framework.model import Model


class BAOModel(Model):

    def __init__(self):
        super().__init__("BAO")
        self.add_param(r"$\Omega_m$", 0.1, 0.6)
        self.add_param(r"$\alpha$", 0.8, 1.2)

    def get_likelihood(self, omega_m, alpha):
        from scipy.stats import norm
        return norm.logpdf(omega_m, 0.3, 0.1) + norm.logpdf(alpha, 1.0, 0.02)

if __name__ == "__main__":
    bao = BAOModel()
