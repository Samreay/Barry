from functools import lru_cache
import numpy as np

from barry.cosmology.pk2xi import PowerToCorrelationGauss
from barry.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth
from barry.models.model import Model
from barry.models import PowerSpectrumFit
from scipy.interpolate import splev, splrep
from scipy import integrate


class CorrelationFunctionFit(Model):
    """ A generic model for computing correlation functions."""

    def __init__(
        self,
        name="BAO Correlation Polynomial Fit",
        smooth_type="hinton2017",
        fix_params=("om"),
        smooth=False,
        correction=None,
        isotropic=True,
    ):
        """Generic correlation function model

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
        super().__init__(name, correction=correction, isotropic=isotropic)
        self.parent = PowerSpectrumFit(fix_params=fix_params, smooth_type=smooth_type, correction=correction, isotropic=isotropic)

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

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.pk2xi_0 = None
        self.pk2xi_2 = None
        self.pk2xi_4 = None

    def set_data(self, data):
        """Sets the models data, including fetching the right cosmology and PT generator.

        Note that if you pass in multiple datas (ie a list with more than one element),
        they need to have the same cosmology.

        Parameters
        ----------
        data : dict, list[dict]
            A list of datas to use
        """
        super().set_data(data)
        self.pk2xi_0 = PowerToCorrelationGauss(self.camb.ks, ell=0)
        self.pk2xi_2 = PowerToCorrelationGauss(self.camb.ks, ell=2)
        self.pk2xi_4 = PowerToCorrelationGauss(self.camb.ks, ell=4)

    def declare_parameters(self):
        """ Defines model parameters, their bounds and default value. """
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        self.add_param("b0", r"$b0$", 0.01, 10.0, 1.0)  # Linear galaxy bias for monopole
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles
            self.add_param("b2", r"$b2$", 0.01, 10.0, 1.0)  # Linear galaxy bias for quadrupole

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """Computes the smoothed linear power spectrum and the wiggle ratio.

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
        res = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(
            self.camb.ks, res["pk_lin"], method=self.smooth_type, om=om, h0=self.camb.h0
        )  # Get the smoothed power spectrum
        pk_ratio = res["pk_lin"] / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

    @lru_cache(maxsize=32)
    def get_alphas(self, alpha, epsilon):
        """Computes values of alpha_par and alpha_perp from the input values of alpha and epsilon

        Parameters
        ----------
        alpha : float
            The isotropic dilation scale
        epsilon: float
            The anisotropic warping

        Returns
        -------
        alpha_par : float
            The dilation scale parallel to the line-of-sight
        alpha_perp : float
            The dilation scale perpendicular to the line-of-sight

        """
        return alpha * (1.0 + epsilon) ** 2, alpha / (1.0 + epsilon)

    @lru_cache(maxsize=32)
    def get_sprimefac(self, epsilon):
        """Computes the prefactor to dilate a s value given epsilon, such that sprime = s * sprimefac * alpha

        Parameters
        ----------
        epsilon: float
            The anisotropic warping

        Returns
        -------
        kprimefac : np.ndarray
            The mu dependent prefactor for dilating a k value

        """
        musq = self.mu ** 2
        epsilonsq = (1.0 + epsilon) ** 2
        sprimefac = np.sqrt(musq * epsilonsq ** 2 + (1.0 - musq) / epsilonsq)
        return sprimefac

    @lru_cache(maxsize=32)
    def get_muprime(self, epsilon):
        """Computes dilated values of mu given input values of epsilon for the correlation function

        Parameters
        ----------
        epsilon: float
            The anisotropic warping

        Returns
        -------
        muprime : np.ndarray
            The dilated mu values

        """
        musq = self.mu ** 2
        muprime = self.mu / np.sqrt(musq + (1.0 - musq) / (1.0 + epsilon) ** 6)
        return muprime

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the dilated correlation function multipoles at distance d given the supplied params

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi0 : np.ndarray
            the model monopole interpolated to sprime.
        xi2 : np.ndarray
            the model quadrupole interpolated to sprime. Will be 'None' if the model is isotropic

        """
        # Generate the power spectrum multipoles at the undilated k-values without shape additions
        ks = self.camb.ks
        kprime, pk0, pk2, pk4 = self.parent.compute_power_spectrum(ks, p, smooth=smooth, shape=False, dilate=False)

        if self.isotropic:
            sprime = p["alpha"] * dist
            xi0 = p["b0"] * self.pk2xi_0.__call__(ks, pk0, sprime)
            xi2 = None
            xi4 = None
        else:
            # Construct the dilated 2D correlation function by splineing the undilated multipoles. We could have computed these
            # directly at sprime, but sprime depends on both s and mu, so splining is probably quicker
            epsilon = np.round(p["epsilon"], decimals=5)
            sprime = np.outer(dist * p["alpha"], self.get_sprimefac(epsilon))
            muprime = self.get_muprime(epsilon)
            xi0 = splev(sprime, splrep(dist, self.pk2xi_0.__call__(ks, pk0, dist)))
            xi2 = splev(sprime, splrep(dist, self.pk2xi_2.__call__(ks, pk2, dist)))
            xi4 = splev(sprime, splrep(dist, self.pk2xi_4.__call__(ks, pk4, dist)))
            xi2d = xi0 + 0.5 * (3.0 * muprime ** 2 - 1) * xi2 + 0.125 * (35.0 * muprime ** 4 - 30.0 * muprime ** 2 + 3.0) * xi4

            xi0 = p["b0"] * integrate.simps(xi2d, self.mu, axis=1)
            xi2 = 3.0 * p["b2"] * integrate.simps(xi2d * self.mu ** 2, self.mu, axis=1)
            xi2 = 2.5 * (xi2 - xi0)

        return sprime, xi0, xi2

    def get_model(self, p, data, smooth=False):
        """Gets the model prediction using the data passed in and parameter location specified

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
            The concatenated xi_{\ell}(s) predictions at the dilated distances given p and data['dist']

        """

        dist, xi0, xi2 = self.compute_correlation_function(data["dist"], p, smooth=smooth)

        if self.isotropic:
            xi_model = xi0
        else:
            xi_model = np.concatenate([xi0, xi2])

        return xi_model

    def get_likelihood(self, p, d):
        """Uses the stated likelihood correction and `get_model` to compute the likelihood

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

        diff = d["xi"] - xi_model
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())
        return self.get_chi2_likelihood(diff, d["icov"], num_mocks=num_mocks, num_params=num_params)

    def plot(self, params, smooth_params=None):
        import matplotlib.pyplot as plt

        ss = self.data[0]["dist"]
        xi0 = self.data[0]["xi0"]
        xi0err = np.sqrt(np.diag(self.data[0]["cov"])[0 : len(ss)])
        xi0mod = self.get_model(params, self.data[0])[0 : len(ss)]
        if not self.isotropic:
            xi2 = self.data[0]["xi2"]
            xi2err = np.sqrt(np.diag(self.data[0]["cov"])[len(ss) :])
            xi2mod = self.get_model(params, self.data[0])[len(ss) :]

        if smooth_params is not None:
            xi0smooth = self.get_model(smooth_params, self.data[0], smooth=True)[: len(ss)]
            if not self.isotropic:
                xi2smooth = self.get_model(smooth_params, self.data[0], smooth=True)[len(ss) :]
        else:
            xi0smooth = self.get_model(params, self.data[0], smooth=True)[: len(ss)]
            if not self.isotropic:
                xi2smooth = self.get_model(params, self.data[0], smooth=True)[len(ss) :]

        def adj(data, err=False):
            if err:
                return data
            else:
                return data - smooth

        fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        axes[0].errorbar(ss, ss * ss * xi0, yerr=ss * ss * xi0err, fmt="o", c="r", ms=4, label=self.data[0]["name"])
        axes[1].errorbar(ss, xi0 - xi0smooth, yerr=xi0err, fmt="o", c="r", ms=4, label=self.data[0]["name"])

        axes[0].plot(ss, ss * ss * xi0mod, color="r", label=self.get_name())
        axes[1].plot(ss, xi0mod - xi0smooth, color="r", label=self.get_name())

        if not self.isotropic:
            axes[0].errorbar(ss, ss * ss * xi2, yerr=ss * ss * xi2err, fmt="o", c="b", ms=4, label=self.data[0]["name"])
            axes[1].errorbar(ss, xi2 - xi2smooth, yerr=xi2err, fmt="o", c="b", ms=4, label=self.data[0]["name"])

            axes[0].plot(ss, ss * ss * xi2mod, color="b", label=self.get_name())
            axes[1].plot(ss, xi2mod - xi2smooth, color="b", label=self.get_name())

        string = f"Likelihood: {self.get_likelihood(params, self.data[0]):0.2f}\n"
        string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
        va = "bottom"
        ypos = 0.02
        axes[0].annotate(string, (0.01, ypos), xycoords="axes fraction", horizontalalignment="left", verticalalignment=va)
        axes[1].legend()
        axes[1].set_xlabel("s")
        if self.postprocess is None:
            axes[1].set_ylabel("xi(s) / xi_{smooth}(s)")
        else:
            axes[1].set_ylabel("xi(s) / data")
        axes[0].set_ylabel("s^2 * xi(s)")
        plt.show()


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_correlation_Beutler2017.py")
    print("bao_correlation_Ding2018.py")
    print("bao_correlation_Seo2016.py")
