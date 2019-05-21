from scipy.interpolate import splev, splrep

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method
from barry.framework.model import Model
import numpy as np

# TODO: make h0 and omega_m more easily changable if we are not fitting them
class PowerSpectrumFit(Model):
    def __init__(self, fit_omega_m=False, smooth_type="hinton2017", name="Base Power Spectrum Fit"):
        super().__init__(name)
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)
        self.fit_omega_m = fit_omega_m

        # Set up data structures for model fitting
        self.h0 = 0.6751
        self.camb = CambGenerator(om_resolution=20, h0=self.h0)
        if not self.fit_omega_m:
            self.omega_m = 0.3121
            self.r_s, self.pk_lin = self.camb.get_data(om=self.omega_m)
            self.pk_smooth_lin = smooth(self.camb.ks, self.pk_lin, method=self.smooth_type, om=self.omega_m, h0=self.h0) # Get the smoothed power spectrum
            self.pk_ratio = (self.pk_lin / self.pk_smooth_lin - 1.0) # Get the ratio

        self.declare_parameters()

    def declare_parameters(self):
        # Define parameters
        if self.fit_omega_m:
            self.add_param("om", r"$\Omega_m$", 0.1, 0.5)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2)  # Stretch
        self.add_param("b", r"$b$", 0.8, 1.2)  # bias

    def compute_basic_power_spectrum(self, p):
        """ Computes the smoothed, linear power spectrum and the wiggle ratio

        Parameters
        ----------
        ks : np.ndarray
            Array of wavenumbers to compute.
            This should probably be camb.ks, not the k values of your data
        p : dict
            dictionary of parameter names to their values

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        if self.fit_omega_m:
            r_s, pk_lin = self.camb.get_data(om=p["om"], h0=self.h0)
            pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=p["om"], h0=self.h0)  # Get the smoothed power spectrum
            pk_ratio = (pk_lin / pk_smooth_lin - 1.0) # Get the ratio
        else:
            pk_smooth_lin, pk_ratio = self.pk_smooth_lin, self.pk_ratio

        return pk_smooth_lin, pk_ratio

    def compute_power_spectrum(self, k, p):
        """ Returns the wiggle ratio interpolated at some k/alpha values. Useful if we only want alpha to modify
            the BAO feature and not the smooth component.

        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute.
        p : dict
            dictionary of parameter names to their values

        Returns
        -------
        array
            pk_final - the ratio (pk_lin / pk_smooth - 1.0),  interpolated to k/alpha.

        """
        ks = self.camb.ks
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum()
        pk_final = splev(k / p["alpha"], splrep(ks, pk_ratio_dewiggled))
        return pk_final

    def adjust_model_window_effects(self, pk_generated):

        p0 = np.sum(self.data["w_scale"] * pk_generated)
        integral_constraint = self.data["w_pk"][2] * p0

        pk_convolved = np.atleast_2d(pk_generated) @ self.data["w_transform"]
        pk_normalised = (pk_convolved - integral_constraint).flatten()
        # Get the subsection of our model which corresponds to the data k values
        return pk_normalised, self.data["w_mask"]

    def get_likelihood(self, p):
        d = self.data
        if not self.fit_omega_m:
            p["om"] = 0.3121

        pk_model = self.get_model(d, p)

        # Compute the chi2
        diff = (d["pk"] - pk_model)
        chi2 = diff.T @ d["icov"] @ diff
        return -0.5 * chi2

    def get_model(self, data, p):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], p)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)
        return pk_windowed[mask]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model = PowerSpectrumFit()

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 5, "b": 0, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

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
