import logging
import numpy as np
from barry.framework.models.bao_correlation import CorrelationPolynomial


class CorrBeutler2017(CorrelationPolynomial):

    def __init__(self, smooth_type="hinton2017", name="Corr Beutler 2017", fix_params=['om'], smooth=False):
        super().__init__(smooth_type, name, fix_params, smooth)
        self.declare_parameters()

    def declare_parameters(self):
        # Define parameters
        super().declare_parameters()
        self.add_param("sigma_nl", r"$\Sigma_{NL}$", 1.0, 20.0, 1.0)  # dampening
        self.add_param("a1", r"$a_1$", -100, 100, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2, 0)  # Polynomial marginalisation 3

    def compute_correlation_function(self, d, p, smooth=False):
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        pk_lin, pk_smooth = self.compute_basic_power_spectrum(p["om"])

        # Blend the two
        if smooth:
            pk_dewiggled = pk_smooth
        else:
            pk_linear_weight = np.exp(-0.5 * (ks * p["sigma_nl"])**2)
            pk_dewiggled = pk_linear_weight * pk_lin + (1 - pk_linear_weight) * pk_smooth

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi.pk2xi(ks, pk_dewiggled, d * p["alpha"])

        # Polynomial shape
        shape = p["a1"] / (d ** 2) + p["a2"] / d + p["a3"]

        # Add poly shape to xi model, include bias correction
        model = xi * p["b"] + shape
        return model


if __name__ == "__main__":
    import sys
    sys.path.append("../../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    bao = CorrelationPolynomial()

    from barry.framework.datasets.mock_correlation import MockSDSSCorrelationFunction
    dataset = MockSDSSCorrelationFunction()
    data = dataset.get_data()
    bao.set_data(data)

    import timeit
    n = 500
    p = {"om":0.3, "alpha":1.0, "sigma_nl": 5.0, "b": 1.0, "a1": 0, "a2": 0, "a3": 0}

    def test():
        bao.get_likelihood(p)
    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ss = data["dist"]
        xi0 = data["xi0"]
        xi = bao.compute_correlation_function(ss, p)
        print(xi0)
        print(xi)
        import matplotlib.pyplot as plt
        plt.errorbar(ss, ss * ss * xi, yerr=ss * ss * np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
        plt.plot(ss, ss * ss * xi0, c='r')
        plt.show()
