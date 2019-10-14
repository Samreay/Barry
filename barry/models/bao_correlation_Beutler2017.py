import logging
import numpy as np
from barry.models.bao_correlation import CorrelationFunctionFit


class CorrBeutler2017(CorrelationFunctionFit):
    """  xi(s) model inspired from Beutler 2017 and Ross 2015.
    """

    def __init__(self, name="Corr Beutler 2017", smooth_type="hinton2017", fix_params=["om"], smooth=False, correction=None):
        super().__init__(name, smooth_type, fix_params, smooth, correction=correction)

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
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum(p["om"])

        # Blend the two
        if smooth:
            pk_dewiggled = pk_smooth
        else:
            pk_linear_weight = np.exp(-0.5 * (ks * p["sigma_nl"]) ** 2)
            pk_dewiggled = (pk_linear_weight * (1 + pk_ratio_dewiggled) + (1 - pk_linear_weight)) * pk_smooth

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi.__call__(ks, pk_dewiggled, d * p["alpha"])

        # Polynomial shape
        shape = p["a1"] / (d ** 2) + p["a2"] / d + p["a3"]

        # Add poly shape to xi model, include bias correction
        model = xi * p["b"] + shape
        return model


if __name__ == "__main__":
    import sys
    import timeit
    from barry.datasets.dataset_correlation_function import CorrelationFunction_SDSS_DR12_Z061_NGC

    sys.path.append("../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    dataset = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=False)
    data = dataset.get_data()
    model_pre = CorrBeutler2017()
    model_pre.set_data(data)

    dataset = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=True)
    data = dataset.get_data()
    model_post = CorrBeutler2017()
    model_post.set_data(data)

    p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 5.0, "sigma_s": 5, "b": 2.0, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    n = 200

    def test_pre():
        model_pre.get_likelihood(p, data[0])

    def test_post():
        model_post.get_likelihood(p, data[0])

    print("Pre-reconstruction likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test_pre, number=n) * 1000 / n))
    print("Post-reconstruction likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test_post, number=n) * 1000 / n))

    if True:
        p, minv = model_pre.optimize()
        print("Pre reconstruction optimisation:")
        print(p)
        print(minv)
        model_pre.plot(p)

        print("Post reconstruction optimisation:")
        p, minv = model_post.optimize()
        print(p)
        print(minv)
        model_post.plot(p)
