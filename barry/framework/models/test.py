import numpy as np
from scipy.stats import norm

from barry.framework.model import Model


class TestModel(Model):

    def __init__(self):
        super().__init__("TestModel")
        self.add_param("om", r"$\Omega_m$", 0.1, 0.6)
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)

    def get_likelihood(self, omega_m, alpha):
        return np.sum(norm.logpdf(self.data, loc=omega_m, scale=alpha))

if __name__ == "__main__":
    bao = TestModel()
