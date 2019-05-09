import logging
import numpy as np
from scipy.interpolate import splev, splrep

from barry.framework.cosmology.bao_extractor import extract_bao
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.power_spectrum_smoothing import smooth_hinton2017
from barry.framework.model import Model


class BAOExtractor(Model):

    def __init__(self, r_s, fit_omega_m=False, name="BAO Extractor Power Spectrum Polynomial Fit"):
        super().__init__(name)

        self.r_s_fiducial = r_s
        # Define parameters
        self.fit_omega_m = fit_omega_m
        if self.fit_omega_m:
            self.add_param("om", r"$\Omega_m$", 0.1, 0.5)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("sigma_nl", r"$\Sigma_{nl}$", 1.0, 20.0)  # dampening
        self.add_param("b", r"$b$", 0.01, 10.0)  # Bias
        # self.add_param("a1", r"$a_1$", -50000.0, 50000.0)  # Polynomial marginalisation 1
        # self.add_param("a2", r"$a_2$", -50000.0, 50000.0)  # Polynomial marginalisation 2
        # self.add_param("a3", r"$a_3$", -50000.0, 50000.0)  # Polynomial marginalisation 3
        # self.add_param("a4", r"$a_4$", -1000.0, 1000.0)  # Polynomial marginalisation 4
        # self.add_param("a5", r"$a_5$", -10.0, 10.0)  # Polynomial marginalisation 5

        # Set up data structures for model fitting
        self.h0 = 0.6751
        self.camb = CambGenerator(h0=self.h0)

        if not self.fit_omega_m:
            self.omega_m = 0.3121
            self.r_s, self.pk_lin = self.camb.get_data(om=self.omega_m)

        self.nice_data = None  # Place to store things like invert cov matrix

    def compute_power_spectrum(self, k, om, alpha, sigma_nl, b):
        """ Computes the correlation function at distance d given the supplied params
        
        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute
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
        a4 : float
            Polynomial shape 4
        a5 : float
            Polynomial shape 5
            
        Returns
        -------
        array
            The correlation function power at the requested distances.
        
        """
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        if self.fit_omega_m:
            _, pk_lin = self.camb.get_data(om=om, h0=self.h0)
        else:
            pk_lin = self.pk_lin

        # Get the smoothed power spectrum
        pk_smooth = smooth_hinton2017(ks, pk_lin)

        # Get the ratio
        pk_ratio = pk_lin / pk_smooth

        # Smooth the ratio based on sigma_nl
        pk_ratio_dewiggled = 1.0 + (pk_ratio - 1) * np.exp(-0.5 * (ks * sigma_nl)**2)

        # Polynomial shape
        #shape = a1 * k + a2 + a3 / k + a4 / (k * k) + a5 / (k ** 3)

        # Combine everything. Weirdly. With lots of spline interpolation.
        pk_final = (splev(k, splrep(ks, b * pk_smooth))) * splev(k / alpha, splrep(ks, pk_ratio_dewiggled))

        return pk_final

    def adjust_model_window_effects(self, pk_generated):

        p0 = np.sum(self.data["w_scale"] * pk_generated)
        integral_constraint = self.data["w_pk"][2] * p0

        pk_convolved = np.atleast_2d(pk_generated) @ self.data["w_transform"]
        pk_normalised = (pk_convolved - integral_constraint).flatten()
        # Get the subsection of our model which corresponds to the data k values
        return pk_normalised, self.data["w_mask"]

    def get_model(self, data, om, alpha, sigma_nl, b):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], om, alpha, sigma_nl, b)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)

        _, pk_model = extract_bao(data["ks_output"], pk_windowed, self.r_s_fiducial)
        pk_final = pk_model[mask]
        return pk_final

    def get_likelihood(self, *params):

        d = self.data
        if self.fit_omega_m:
            om, alpha, sigma_nl, b = params
        else:
            alpha, sigma_nl, b = params
            om = 0.3121

        pk_model = self.get_model(d, om, alpha, sigma_nl, b)

        # Compute the chi2
        diff = (d["pk"] - pk_model)
        chi2 = diff.T @ d["icov"] @ diff
        return -0.5 * chi2


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")

    c = CambGenerator()
    r_s, _ = c.get_data()
    bao = BAOExtractor(r_s, fit_omega_m=True)

    from barry.framework.datasets.mock_bao_extractor import MockBAOExtractorPowerSpectrum
    dataset = MockBAOExtractorPowerSpectrum(r_s, step_size=2)
    data = dataset.get_data()
    bao.set_data(data)

    # print(bao.get_likelihood(0.3, 1.0, 5.0, 1.0, 0, 0, 0, 0, 0))

    # import timeit
    # n = 5
    # def test():
    #     bao.get_likelihood(0.3, 1.0, 5.0, 1.0, 0, 0, 0, 0, 0)
    # print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = bao.get_model(data, 0.3, 1, 5, 1)
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k', label="Data")
        plt.plot(ks, pk2, '-', c='r', label="Model")
        plt.legend()
        plt.show()
