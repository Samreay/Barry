import logging
import numpy as np
from scipy.interpolate import splev, splrep
import sys
sys.path.append("../../..")
from barry.framework.models.bao_power import PowerSpectrumFit


class PowerPolynomial(PowerSpectrumFit):

    def __init__(self, fit_omega_m=False, smooth_type="hinton2017", name="BAO Power Spectrum Polynomial Fit"):
        super().__init__(fit_omega_m=fit_omega_m, smooth_type=smooth_type, name=name)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("b", r"$b$", 0.01, 10.0)  # Bias
        self.add_param("a1", r"$a_1$", -50000.0, 50000.0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -50000.0, 50000.0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -50000.0, 50000.0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -1000.0, 1000.0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -10.0, 10.0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p):
        """ Computes the correlation function at distance d given the supplied params
        
        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute
        p : dict
            dictionary of parameter names to their values
            
        Returns
        -------
        array
            The correlation function power at the requested distances.
        
        """
        ks = self.camb.ks
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum(ks, p)

        # Polynomial shape
        shape = p["a1"] * k + p["a2"] + p["a3"] / k + p["a4"] / (k * k) + p["a5"] / (k ** 3)

        # Combine everything. Weirdly. With lots of spline interpolation.
        pk_final = (splev(k, splrep(ks, p["b"] * pk_smooth)) + shape) * splev(k / p["alpha"], splrep(ks, pk_ratio_dewiggled))

        return pk_final

    def get_model(self, data, p):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], p)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)
        return pk_windowed[mask]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model = PowerPolynomial(fit_omega_m=True)

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 5, "b": 1, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    import timeit
    n = 500

    def test():
        model.get_likelihood(p)

    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model.get_model(data, p)
        model.smooth_type = "eh1998"
        pk3 = model.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
        plt.plot(ks, pk2, '.', c='r')
        plt.plot(ks, pk3, '+', c='b')
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.show()
