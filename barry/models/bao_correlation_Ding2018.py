import logging
import numpy as np

from barry.models import PowerDing2018
from barry.models.bao_correlation import CorrelationFunctionFit


class CorrDing2018(CorrelationFunctionFit):
    """ xi(s) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(self, name="Corr Ding 2018", recon=False, smooth_type="hinton2017", fix_params=("om", "f"), smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, smooth=smooth, correction=correction)
        self.parent = PowerDing2018(fix_params=fix_params, smooth_type=smooth_type, recon=recon, correction=correction)

    def set_data(self, data):
        super().set_data(data)
        self.parent.set_data(data)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0, 5.0)  # Non-linear galaxy bias
        self.add_param("a1", r"$a_1$", -100, 100, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2, 0)  # Polynomial marginalisation 3

    def compute_correlation_function(self, d, p, smooth=False):
        """ Computes the correlation function model at d*alpha using the Ding et. al., 2018 EFT0 model

        Parameters
        ----------
        d : np.ndarray
            Array of separations to compute
        p : dict
            dictionary of parameter names to their values

        Returns
        -------
        array
            xi_final - The correlation function at the dilated d-values

        """

        # Get the basic power spectrum components
        ks, pk1d = self.parent.compute_power_spectrum(p, smooth=smooth, shape=False)

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi(ks, pk1d, d * p["alpha"])

        # Polynomial shape
        shape = p["a1"] / (d ** 2) + p["a2"] / d + p["a3"]

        # Add poly shape to xi model, include bias correction
        model = xi + shape
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
    model_pre = CorrDing2018(recon=False)
    model_pre.set_data(data)

    dataset = CorrelationFunction_SDSS_DR12_Z061_NGC(recon=True)
    data = dataset.get_data()
    model_post = CorrDing2018(recon=True)
    model_post.set_data(data)

    p = {"om": 0.3, "alpha": 1.0, "sigma_s": 10.0, "b": 1.6, "b_delta": 1, "a1": 0, "a2": 0, "a3": 0}

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
