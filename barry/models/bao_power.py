from functools import lru_cache

from scipy import integrate
from scipy.interpolate import splev, splrep

from barry.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method
from barry.models.model import Model
import numpy as np


class PowerSpectrumFit(Model):
    """ Generic power spectrum model """

    def __init__(self, name="Pk Basic", smooth_type="hinton2017", fix_params=("om"), postprocess=None, smooth=False, correction=None, isotropic=True):
        """ Generic power spectrum function model

        Parameters
        ----------
        name : str, optional
            Name of the model
        smooth_type : str, optional
            The sort of smoothing to use. Either 'hinton2017' or 'eh1998'
        fix_params : list[str], optional
            Parameter names to fix to their defaults. Defaults to just `[om]`.
        postprocess : `Postprocess` object
            The class to postprocess model predictions. Defaults to none.
        smooth : bool, optional
            Whether to generate a smooth model without the BAO feature. Defaults to `false`.
        correction : `Correction` enum.
            Defaults to `Correction.SELLENTIN
        """
        super().__init__(name, postprocess=postprocess, correction=correction, isotropic=isotropic)
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.smooth = smooth

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

    def declare_parameters(self):
        """ Defines model parameters, their bounds and default value. """
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        self.add_param("b", r"$b$", 0.1, 8.0, 1.73)  # bias (applied to 2D power, so same for all multipoles)
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed linear power spectrum and the wiggle ratio

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        pk_smooth_lin : np.ndarray
            The power spectrum smoothed out
        pk_ratio : np.ndarray
            the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        res = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(self.camb.ks, res["pk_lin"], method=self.smooth_type, om=om, h0=self.camb.h0)  # Get the smoothed power spectrum
        pk_ratio = res["pk_lin"] / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

    @lru_cache(maxsize=32)
    def get_alphas(self, alpha, epsilon):
        """ Computes values of alpha_par and alpha_perp from the input values of alpha and epsilon

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
    def get_kprimefac(self, epsilon):
        """ Computes the prefactor to dilate a k value given epsilon, such that kprime = k * kprimefac / alpha

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
        kprimefac = np.sqrt(musq / epsilonsq ** 2 + (1.0 - musq) * epsilonsq)
        return kprimefac

    @lru_cache(maxsize=32)
    def get_muprime(self, epsilon):
        """ Computes dilated values of mu given input values of epsilon for the power spectrum

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
        muprime = self.mu / np.sqrt(musq + (1.0 + epsilon) ** 6 * (1.0 - musq))
        return muprime

    def compute_power_spectrum(self, k, p, smooth=False, shape=True, dilate=True, data_name=None):
        """ Get raw ks and p(k) multipoles for a given parametrisation dilated based on the values of alpha and epsilon

        Parameters
        ----------
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature
        shape : bool, optional
            Whether or not to include shape marginalisation terms.


        Returns
        -------
        ks : np.ndarray
            Wavenumbers of the computed pk
        pk0 : np.ndarray
            the model monopole interpolated using the dilation scales.
        pk2 : np.ndarray
            the model quadrupole interpolated using the dilation scales. Will be 'None' if the model is isotropic

        """
        ks = self.camb.ks
        pk_smooth, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # Work out the dilated values for the power spectra
        if self.isotropic:
            if dilate:
                kprime = k / p["alpha"]
            else:
                kprime = k
        else:
            if dilate:
                epsilon = p["epsilon"]
                kprime = np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
                muprime = self.get_muprime(epsilon)
            else:
                kprime = np.tile(k, (self.nmu, 1))
                muprime = self.mu

        if smooth:
            pkprime = p["b"] ** 2 * splev(kprime, splrep(ks, pk_smooth))
        else:
            pkprime = p["b"] ** 2 * splev(kprime, splrep(ks, pk_smooth * (1 + pk_ratio)))

        # Get the multipoles
        if self.isotropic:
            pk0 = pkprime
            pk2 = None
            pk4 = None
        else:
            growth = p["f"]
            s = self.camb.smoothing_kernel
            kaiser_prefac = 1.0 + growth / p["b"] * np.outer(1.0 - s, muprime ** 2)
            pk2d = kaiser_prefac * pkprime

            pk0 = integrate.simps(pk2d, self.mu, axis=1)
            pk2 = 3.0 * integrate.simps(pk2d * self.mu ** 2, self.mu, axis=1)
            pk4 = 1.125 * (35.0 * integrate.simps(pk2d * self.mu ** 4, self.mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0)

        return kprime, pk0, pk2, pk4

    def adjust_model_window_effects(self, pk_generated, data, window=True):
        """ Take the window effects into account.

        Parameters
        ----------
        pk_generated : np.ndarray
            The p(k) values generated at the window function input ks
        data : dict
            The data dictionary containing the window scale `w_scale`,
            transformation matrix `w_transform`, integral constraint `w_pk`
            and mask `w_mask`

        Returns
        -------
        pk_normalised : np.ndarray
            The transformed, corrected power spectrum
        mask : np.ndarray
            A boolean mask used for selecting the final data out of pk_normalised.
            Mask is not applied in this function because for the BAO extractor
            post processing we want to take the powers outside the mask into account
            and *then* mask.
        """

        if self.isotropic:
            # TODO: For isotropic, incorporate the integral constraint into the window function matrix when the data is pickled
            # to make the window function correction more similar between this and anisotropic.
            if window:
                p0 = np.sum(data["w_scale"] * pk_generated)
                integral_constraint = data["w_pk"] * p0

                pk_convolved = np.atleast_2d(pk_generated) @ data["w_transform"]
                pk_normalised = (pk_convolved - integral_constraint).flatten()
            else:
                pk_normalised = splev(data["ks_output"], splrep(data["ks_input"], pk_generated))

        else:
            # Multiply the model by the M matrix to get the 5 multipoles
            pk_mod = data["m_transform"] @ pk_generated

            if window:
                # Convolve the model
                pk_normalised = data["w_transform"] @ pk_mod
            else:
                pk_normalised = []
                for i in range(5):
                    pk_normalised.append(
                        splev(data["ks_output"], splrep(data["ks_input"], pk_mod[i * len(data["ks_input"]) : (i + 1) * len(data["ks_input"])]))
                    )
                pk_normalised = np.array(pk_normalised).flatten()

        return pk_normalised, data["w_mask"]

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
        pk_model = self.get_model(p, d, smooth=self.smooth)
        pk_model_fit = np.concatenate([pk_model[: len(d["ks"])], pk_model[2 * len(d["ks"]) : 3 * len(d["ks"])], pk_model[4 * len(d["ks"]) : 5 * len(d["ks"])]])
        # pk_model_fit = pk_model

        # Compute the chi2
        diff = d["pk"] - pk_model_fit
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())
        return self.get_chi2_likelihood(diff, d["icov"], num_mocks=num_mocks, num_params=num_params)

    def get_model(self, p, d, smooth=False):
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
        data_name : str, optional
            The name used to access precomputed values.

        Returns
        -------
        pk_model : np.ndarray
            The p(k) predictions given p and data, k values correspond to d['ks_output']

        """

        ks, pk0, pk2, pk4 = self.compute_power_spectrum(d["ks_input"], p, smooth=smooth, data_name=d["name"])

        # Morph it into a model representative of our survey and its selection/window/binning effects
        if self.isotropic:
            pk_model, mask = self.adjust_model_window_effects(pk0, d)
            if self.postprocess is not None:
                pk_model = self.postprocess(ks=d["ks_output"], pk=pk_model, mask=mask)
            else:
                pk_model = pk_model[mask]
        else:
            pk_generated = np.concatenate([pk0, pk2, pk4])
            pk_model, mask = self.adjust_model_window_effects(pk_generated, d, window=True)
            pk_model = pk_model[mask]

        return pk_model

    def plot(self, params, smooth_params=None):
        import matplotlib.pyplot as plt

        ks = self.data[0]["ks"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        mod = self.get_model(params, self.data[0])

        if smooth_params is not None:
            smooth = self.get_model(smooth_params, self.data[0], smooth=True)
        else:
            smooth = self.get_model(params, self.data[0], smooth=True)

        fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        axes[0].errorbar(ks, ks * self.data[0]["pk0"], yerr=ks * err[: len(ks)], fmt="o", c="k", ms=4, label=r"$P_{0}(k)$")
        axes[1].errorbar(ks, self.data[0]["pk0"] / smooth[: len(ks)], yerr=(err / smooth)[: len(ks)], fmt="o", c="k", ms=4)
        axes[0].plot(ks, ks * mod[: len(ks)], c="k")
        axes[1].plot(ks, (mod / smooth)[: len(ks)], c="k")

        string = f"Likelihood: {self.get_likelihood(params, self.data[0]):0.2f}\n"
        string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
        va = "center" if self.postprocess is None else "top"
        ypos = 0.5 if self.postprocess is None else 0.98
        axes[0].annotate(string, (0.98, ypos), xycoords="axes fraction", horizontalalignment="right", verticalalignment=va)
        axes[1].legend()
        axes[1].set_xlabel("k")
        if self.postprocess is None:
            axes[1].set_ylabel("P(k) / P_{smooth}(k)")
        else:
            axes[1].set_ylabel("P(k) / data")
        axes[0].set_ylabel("k * P(k)")

        axes[0].set_title(self.data[0]["name"] + " + " + self.get_name())

        if not self.isotropic:
            axes[0].errorbar(ks, ks * self.data[0]["pk2"], yerr=ks * err[2 * len(ks) : 3 * len(ks)], fmt="o", c="r", ms=4, label=r"$P_{2}(k)$")
            axes[1].errorbar(ks, self.data[0]["pk2"] / smooth[2 * len(ks) : 3 * len(ks)], yerr=(err / smooth)[2 * len(ks) : 3 * len(ks)], fmt="o", c="r", ms=4)
            axes[0].plot(ks, ks * mod[2 * len(ks) : 3 * len(ks)], c="r")
            axes[1].plot(ks, (mod / smooth)[2 * len(ks) : 3 * len(ks)], c="r")

            axes[0].errorbar(ks, ks * self.data[0]["pk4"], yerr=ks * err[4 * len(ks) :], fmt="o", c="b", ms=4, label=r"$P_{4}(k)$")
            # axes[1].errorbar(ks, self.data[0]["pk4"] / smooth[4 * len(ks) :], yerr=(err / smooth)[4 * len(ks) :], fmt="o", c="b", ms=4)
            axes[0].plot(ks, ks * mod[4 * len(ks) :], c="b")
            # axes[1].plot(ks, (mod / smooth)[4 * len(ks) :], c="b")

        plt.legend()

        if not self.isotropic:

            fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

            axes[0].errorbar(ks, ks * self.data[0]["pk1"], yerr=ks * err[1 * len(ks) : 2 * len(ks)], fmt="o", c="r", ms=4, label=r"$P_{1}(k)$")
            axes[1].errorbar(ks, self.data[0]["pk1"] - smooth[1 * len(ks) : 2 * len(ks)], yerr=(err)[1 * len(ks) : 2 * len(ks)], fmt="o", c="r", ms=4)
            axes[0].plot(ks, ks * mod[1 * len(ks) : 2 * len(ks)], c="r")
            axes[1].plot(ks, (mod - smooth)[1 * len(ks) : 2 * len(ks)], c="r")

            axes[0].errorbar(ks, ks * self.data[0]["pk3"], yerr=ks * err[3 * len(ks) : 4 * len(ks)], fmt="o", c="b", ms=4, label=r"$P_{3}(k)$")
            axes[1].errorbar(ks, self.data[0]["pk3"] - smooth[3 * len(ks) : 4 * len(ks)], yerr=(err)[3 * len(ks) : 4 * len(ks)], fmt="o", c="b", ms=4)
            axes[0].plot(ks, ks * mod[3 * len(ks) : 4 * len(ks)], c="b")
            axes[1].plot(ks, (mod - smooth)[3 * len(ks) : 4 * len(ks)], c="b")

            string = f"Likelihood: {self.get_likelihood(params, self.data[0]):0.2f}\n"
            string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
            va = "center" if self.postprocess is None else "top"
            ypos = 0.5 if self.postprocess is None else 0.98
            axes[0].annotate(string, (0.98, ypos), xycoords="axes fraction", horizontalalignment="right", verticalalignment=va)
            axes[1].legend()
            axes[1].set_xlabel("k")
            if self.postprocess is None:
                axes[1].set_ylabel("P(k) - P_{smooth}(k)")
            else:
                axes[1].set_ylabel("P(k) - data")
            axes[0].set_ylabel("k * Im[P(k)]")

            axes[0].set_title(self.data[0]["name"] + " + " + self.get_name())
            plt.legend()

        plt.show()


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_power_Beutler2017.py")
    print("bao_power_Ding2018.py")
    print("bao_power_Noda2019.py")
    print("bao_power_Seo2016.py")
