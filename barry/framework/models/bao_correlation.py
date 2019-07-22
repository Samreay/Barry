from functools import lru_cache
import numpy as np
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.pk2xi import PowerToCorrelationGauss
from barry.framework.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth
from barry.framework.model import Model


class CorrelationPolynomial(Model):
    def __init__(self, smooth_type="hinton2017", name="BAO Correlation Polynomial Fit", fix_params=['om'], smooth=False):
        super().__init__(name)

        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.camb = CambGenerator()
        self.h0 = self.camb.h0
        self.smooth = smooth
        self.pk2xi = PowerToCorrelationGauss(self.camb.ks)

    def declare_parameters(self):
        # Define parameters
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.3121)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch
        self.add_param("b", r"$b$", 0.01, 10.0, 1.0)  # Bias

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed, linear power spectrum and the wiggle ratio

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        r_s, pk_lin = self.camb.get_data(om=om, h0=self.h0)
        pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=om, h0=self.h0)  # Get the smoothed power spectrum
        return pk_lin, pk_smooth_lin

    def compute_correlation_function(self, d, p, smooth=False):
        """ Computes the correlation function at distance d given the supplied params

        Parameters
        ----------
        d : array
            Array of distances in the correlation function to compute
        params : dict
            dictionary of parameter name to float value pairs

        Returns
        -------
        array
            The correlation function power at the requested distances.

        """
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        pk_lin, pk_smooth = self.compute_basic_power_spectrum(p["om"])

        xi = self.pk2xi.pk2xi(ks, pk_lin, d * p["alpha"])
        return xi * p["b"]

    def get_model(self, p, smooth=False):
        pk_model = self.compute_correlation_function(self.data["dist"], p, smooth=smooth)
        return pk_model

    def get_likelihood(self, p):
        d = self.data
        xi_model = self.get_model(p, smooth=self.smooth)

        diff = (d["xi"] - xi_model)
        chi2 = diff.T @ d["icov"] @ diff
        return -0.5 * chi2

    def plot(self, params, smooth_params=None):
        import matplotlib.pyplot as plt

        ss = self.data["dist"]
        xi = self.data["xi0"]
        err = np.sqrt(np.diag(self.data["cov"]))
        xi2 = self.get_model(params)

        if smooth_params is not None:
            smooth = self.get_model(smooth_params, smooth=True)
        else:
            smooth = self.get_model(params, smooth=True)

        def adj(data, err=False):
            if err:
                return data
            else:
                return data - xi

        fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        axes[0].errorbar(ss, ss * ss * xi, yerr=ss * ss * err, fmt="o", c='k', ms=4, label=self.data["name"])
        axes[1].errorbar(ss, adj(xi), yerr=adj(err, err=True), fmt="o", c='k', ms=4, label=self.data["name"])

        axes[0].plot(ss, ss * ss * xi2, label=self.get_name())
        axes[1].plot(ss, adj(xi2), label=self.get_name())

        string = f"Likelihood: {self.get_likelihood(params):0.2f}\n"
        string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
        va = "bottom"
        ypos = 0.02
        axes[0].annotate(string, (0.01, ypos), xycoords="axes fraction", horizontalalignment="left",
                         verticalalignment=va)
        axes[1].legend()
        axes[1].set_xlabel("k")
        if self.postprocess is None:
            axes[1].set_ylabel("xi(s) / xi_{smooth}(s)")
        else:
            axes[1].set_ylabel("xi(s) / data")
        axes[0].set_ylabel("s^2 * xi(s)")
        plt.show()
