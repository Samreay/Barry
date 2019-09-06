from functools import lru_cache
import numpy as np

from barry.cosmology.PT_generator import getCambGeneratorAndPT
from barry.cosmology.pk2xi import PowerToCorrelationGauss
from barry.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth
from barry.models.model import Model


class CorrelationFunctionFit(Model):
    """ A generic model for computing correlation functions."""
    def __init__(self, name="BAO Correlation Polynomial Fit", smooth_type="hinton2017", fix_params=['om'], smooth=False, correction=None):
        """ Generic correlation function model

        Parameters
        ----------
        name : str, optional
            Name of the model
        smooth_type : str, optional
            The sort of smoothing to use. Either 'hinton2017' or 'eh1998'
        fix_params : list[str], optional
            Parameter names to fix to their defaults. Defaults to just `[om]`.
        smooth : bool, optional
            Whether to generate a smooth model without the BAO feature. Defaults to `false`.
        correction : `Correction` enum. Defaults to `Correction.SELLENTIN
        """
        super().__init__(name, correction=correction)

        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.smooth = smooth
        self.camb = None
        self.PT = None
        self.pk2xi = None
        self.recon_smoothing_scale = None
        self.cosmology = None

    def set_data(self, data):
        """ Sets the models data, including fetching the right cosmology and PT generator.

        Note that if you pass in multiple datas (ie a list with more than one element),
        they need to have the same cosmology.

        Parameters
        ----------
        data : dict, list[dict]
            A list of datas to use
        """
        super().set_data(data)
        c = data[0]["cosmology"]
        if self.cosmology != c:
            self.recon_smoothing_scale = c["reconsmoothscale"]
            self.camb, self.PT = getCambGeneratorAndPT(h0=c["h0"], ob=c["ob"], redshift=c["z"], ns=c["ns"], smooth_type=self.smooth_type, recon_smoothing_scale=self.recon_smoothing_scale)
            self.pk2xi = PowerToCorrelationGauss(self.camb.ks)
            self.set_default("om", c["om"])

    def declare_parameters(self):
        """ Defines model parameters, their bounds and default value. """
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch
        self.add_param("b", r"$b$", 0.01, 10.0, 1.0)  # Linear galaxy bias

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed linear power spectrum and the wiggle ratio.

        Uses a fixed h0 as determined by the dataset cosmology.

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth

        """
        # Get base linear power spectrum from camb
        r_s, pk_lin = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=om, h0=self.camb.h0)  # Get the smoothed power spectrum
        pk_ratio = (pk_lin / pk_smooth_lin - 1.0)  # Get the ratio
        return pk_smooth_lin, pk_ratio

    def compute_correlation_function(self, dist, p, smooth=False):
        """ Computes the correlation function at distance d given the supplied params

        Parameters
        ----------
        dist : array
            Array of distances in the correlation function to compute
        params : dict
            dictionary of parameter name to float value pairs

        Returns
        -------
        xi(dist) : np.array
            The correlation function power at the requested distances.

        """
        # Get base linear power spectrum from camb
        ks = self.camb.ks
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum(p["om"])

        # Convert to real space from Fourier space
        xi = self.pk2xi.pk2xi(ks, pk_smooth * (1 + pk_ratio_dewiggled), dist * p["alpha"])
        return xi * p["b"]

    def get_model(self, p, data, smooth=False):
        """ Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        data : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.
        smooth : bool, optional
            Whether to only generate a smooth model without the BAO feature

        Returns
        -------
        xi_model : np.ndarray
            The xi(s) predictions given p and data['dist']

        """
        xi_model = self.compute_correlation_function(data["dist"], p, smooth=smooth)
        return xi_model

    def get_likelihood(self, p, d):
        """ Uses the stated likelihood correction and `get_model` to compute the likelihood

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.

        Returns
        -------
        log_likelihood : float
            The corrected log likelihood
        """

        xi_model = self.get_model(p, d, smooth=self.smooth)

        diff = (d["xi0"] - xi_model)
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())
        return self.get_chi2_likelihood(diff, d["icov"], num_mocks=num_mocks, num_params=num_params)

    def plot(self, params, smooth_params=None):
        import matplotlib.pyplot as plt

        ss = self.data[0]["dist"]
        xi = self.data[0]["xi0"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        xi2 = self.get_model(params, self.data[0])

        if smooth_params is not None:
            smooth = self.get_model(smooth_params, self.data[0], smooth=True)
        else:
            smooth = self.get_model(params, self.data[0], smooth=True)

        def adj(data, err=False):
            if err:
                return data
            else:
                return data - smooth

        fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        axes[0].errorbar(ss, ss * ss * xi, yerr=ss * ss * err, fmt="o", c='k', ms=4, label=self.data[0]["name"])
        axes[1].errorbar(ss, adj(xi), yerr=adj(err, err=True), fmt="o", c='k', ms=4, label=self.data[0]["name"])

        axes[0].plot(ss, ss * ss * xi2, label=self.get_name())
        axes[1].plot(ss, adj(xi2), label=self.get_name())

        string = f"Likelihood: {self.get_likelihood(params, self.data[0]):0.2f}\n"
        string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
        va = "bottom"
        ypos = 0.02
        axes[0].annotate(string, (0.01, ypos), xycoords="axes fraction", horizontalalignment="left",
                         verticalalignment=va)
        axes[1].legend()
        axes[1].set_xlabel("s")
        if self.postprocess is None:
            axes[1].set_ylabel("xi(s) / xi_{smooth}(s)")
        else:
            axes[1].set_ylabel("xi(s) / data")
        axes[0].set_ylabel("s^2 * xi(s)")
        plt.show()
