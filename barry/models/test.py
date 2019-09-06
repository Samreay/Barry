import numpy as np
from scipy.stats import norm

from barry.models.model import Model


class TestModel(Model):
    def __init__(self):
        super().__init__("TestModel")
        self.add_param("om", r"$\Omega_m$", 0.1, 0.6, 0.3)
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)

    def get_likelihood(self, p, d):
        return np.sum(norm.logpdf(d["data"], loc=p["om"], scale=p["alpha"]))

    def plot(self, params, smooth_params=None):
        pass


if __name__ == "__main__":
    bao = TestModel()
