from functools import lru_cache

from scipy import integrate
from scipy.integrate import simps
from scipy.interpolate import splev, splrep

from barry.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method
from barry.models.model import Model
import numpy as np

from barry.utils import break_vector_and_get_blocks


class PowerSpectrumFit(Model):
    """ Generic power spectrum model """

    def __init__(
        self,
        name="Pk Basic",
        smooth_type="hinton2017",
        fix_params=("om"),
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
    ):
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
        super().__init__(name, postprocess=postprocess, correction=correction, isotropic=isotropic, marg=marg)
        self.poly_poles = poly_poles
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
        self.add_param("b", r"$b$", 0.1, 10.0, 1.0)  # Galaxy bias
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
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
        pk_smooth_lin = smooth(
            self.camb.ks, res["pk_lin"], method=self.smooth_type, om=om, h0=self.camb.h0
        )  # Get the smoothed power spectrum
        pk_ratio = res["pk_lin"] / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

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

    def integrate_mu(self, pk2d, isotropic=False):
        pk0 = simps(pk2d, self.mu, axis=1)
        if isotropic:
            pk2 = None
            pk4 = None
        else:
            pk2 = 3.0 * simps(pk2d * self.mu ** 2, self.mu)
            pk4 = 1.125 * (35.0 * simps(pk2d * self.mu ** 4, self.mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0)
        return pk0, pk2, pk4

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False):
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
            kprime = k if for_corr else k / p["alpha"]
        else:
            epsilon = 0 if for_corr else p["epsilon"]
            kprime = np.tile(k, (self.nmu, 1)) if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.mu if for_corr else self.get_muprime(epsilon)

        if smooth:
            pkprime = p["b"] * splev(kprime, splrep(ks, pk_smooth))
        else:
            pkprime = p["b"] * splev(kprime, splrep(ks, pk_smooth * (1 + pk_ratio)))

        # Get the multipoles
        if self.isotropic:
            pk0 = pkprime
            pk2 = None
            pk4 = None
        else:
            growth = p["f"]
            s = self.camb.smoothing_kernel
            kaiser_prefac = 1.0 + growth / np.sqrt(p["b"]) * np.outer(1.0 - s, muprime ** 2)
            pk2d = kaiser_prefac * pkprime

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

        return kprime, pk0, pk2, pk4, np.zeros((1, len(k)))

    def adjust_model_window_effects(self, pk_generated, data, window=True, wide_angle=True):
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

            if window:
                if wide_angle:
                    # Convolve the model and apply wide-angle effects to compute odd multipoles
                    pk_normalised = data["w_m_transform"] @ pk_generated
                else:
                    # Only convolve then model, but don't compute odd multipoles
                    pk_normalised = data["w_transform"] @ pk_generated
            else:
                if wide_angle:
                    # Compute odd multipoles, but no window function convolution
                    pk_normalised = data["m_transform"] @ pk_generated
                else:
                    # Just interpolate the models to the values of ks_output without convolution or wide angle effects
                    pk_normalised = []
                    for i in range(len(data["poles"])):
                        pk_normalised.append(
                            splev(
                                data["ks_output"],
                                splrep(data["ks_input"], pk_generated[i * len(data["ks_input"]) : (i + 1) * len(data["ks_input"])]),
                            )
                        )
                    pk_normalised = np.array(pk_normalised).flatten()

        return pk_normalised

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
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())

        pk_model, pk_model_odd, poly_model, poly_model_odd, mask = self.get_model(p, d, smooth=self.smooth, data_name=d["name"])

        if self.isotropic or d["icov_m_w"][0] is None:
            pk_model, pk_model_odd = pk_model[mask], pk_model_odd[mask]

        if self.marg:
            if self.isotropic or d["icov_m_w"][0] is None:
                len_poly = len(d["ks"]) if self.isotropic else len(d["ks"]) * len(d["fit_poles"])
                poly_model_fit = np.zeros((np.shape(poly_model)[0], len_poly))
                poly_model_fit_odd = np.zeros((np.shape(poly_model)[0], len_poly))
                for n in range(np.shape(poly_model)[0]):
                    poly_model_fit[n], poly_model_fit_odd[n] = poly_model[n, mask], poly_model_odd[n, mask]
            else:
                poly_model_fit, poly_model_fit_odd = poly_model, poly_model_odd

        if self.marg_type == "partial":
            return self.get_chi2_partial_marg_likelihood(
                d["pk"],
                pk_model,
                pk_model_odd,
                poly_model_fit,
                poly_model_fit_odd,
                d["icov"],
                d["icov_m_w"],
                num_mocks=num_mocks,
                num_params=num_params,
            )
        elif self.marg_type == "full":
            return self.get_chi2_marg_likelihood(
                d["pk"], pk_model, pk_model_odd, poly_model_fit, poly_model_fit_odd, d["icov"], d["icov_m_w"], num_mocks=num_mocks
            )
        else:
            return self.get_chi2_likelihood(
                d["pk"], pk_model, pk_model_odd, d["icov"], d["icov_m_w"], num_mocks=num_mocks, num_params=num_params
            )

    def get_model(self, p, d, smooth=False, data_name=None):
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
        poly_model : np.ndarray
            the functions describing any polynomial terms, used for analytical marginalisation
            k values correspond to d['ks_output']
        """

        # Morph it into a model representative of our survey and its selection/window/binning effects
        ks, pks, poly = self.compute_power_spectrum(d["ks_input"], p, smooth=smooth, data_name=data_name)

        # Split the model into even and odd components
        pk_generated = pks[0] if self.isotropic else np.concatenate([pks[0], pks[2]])
        if 4 in d["poles"] and not self.isotropic:
            pk_generated = np.concatenate([pk_generated, pks[4]])

        if self.isotropic:
            pk_model, mask = self.adjust_model_window_effects(pk_generated, d), d["m_w_mask"]
            if self.postprocess is not None:
                pk_model = self.postprocess(ks=d["ks_output"], pk=pk_model, mask=mask)
            pk_model_odd = np.zeros(len(d["ks_output"]))
        else:
            pk_model_odd = np.zeros(5 * len(ks))
            pk_model_odd[len(ks) : 2 * len(ks)] += pks[1]
            pk_model_odd[3 * len(ks) : 4 * len(ks)] += pks[3]
            if d["icov_m_w"][0] is None:
                pk_model, mask = self.adjust_model_window_effects(pk_generated, d), d["m_w_mask"]
                pk_model_odd = self.adjust_model_window_effects(pk_model_odd, d, wide_angle=False)
            else:
                pk_model, mask = pk_generated, np.ones(len(pk_generated), dtype=bool)

        poly_model, poly_model_odd = None, None
        if self.marg:
            n_poly_fac = np.shape(poly)[1] if d["icov_m_w"][0] is None else np.shape(poly)[1] - 2
            len_poly_k = len(d["ks_output"]) if d["icov_m_w"][0] is None else len(d["ks_input"])
            len_poly = len(d["ks_output"]) if self.isotropic else len_poly_k * n_poly_fac
            len_poly_odd = len(d["ks_output"]) if self.isotropic else len_poly_k * np.shape(poly)[1]
            poly_model = np.zeros((np.shape(poly)[0], len_poly))
            poly_model_odd = np.zeros((np.shape(poly)[0], len_poly_odd))
            for n in range(np.shape(poly)[0]):
                poly_generated = poly[n] if self.isotropic else np.concatenate([poly[n, 0], poly[n, 2]])
                if 4 in d["poles"] and not self.isotropic:
                    poly_generated = np.concatenate([poly_generated, poly[n, 4]])
                if self.isotropic:
                    poly_model_long = self.adjust_model_window_effects(poly_generated, d)
                    if self.postprocess is not None:
                        poly_model_long = self.postprocess(ks=d["ks_output"], pk=poly_model_long, mask=mask)
                    poly_model[n] = poly_model_long
                else:
                    poly_generated_odd = np.zeros(5 * len(ks))
                    poly_generated_odd[len(ks) : 2 * len(ks)] += poly[n, 1]
                    poly_generated_odd[3 * len(ks) : 4 * len(ks)] += poly[n, 3]
                    if d["icov_m_w"][0] is None:
                        poly_model[n] = self.adjust_model_window_effects(poly_generated, d)
                        poly_model_odd[n] = self.adjust_model_window_effects(poly_generated_odd, d, wide_angle=False)
                    else:
                        poly_model[n], poly_model_odd[n] = poly_generated, poly_generated_odd

        return pk_model, pk_model_odd, poly_model, poly_model_odd, mask

    def plot(self, params, smooth_params=None, figname=None):
        self.logger.info("Create plot")
        import matplotlib.pyplot as plt

        # Ensures we plot the window convolved model
        icov_m_w = self.data[0]["icov_m_w"]
        self.data[0]["icov_m_w"][0] = None

        ks = self.data[0]["ks"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        mod, mod_odd, polymod, polymod_odd, _ = self.get_model(params, self.data[0], data_name=self.data[0]["name"])
        if smooth_params is not None:
            smooth, smooth_odd, polysmooth, polysmooth_odd, _ = self.get_model(
                smooth_params, self.data[0], smooth=True, data_name=self.data[0]["name"]
            )
        else:
            smooth, smooth_odd, polysmooth, polysmooth_odd, _ = self.get_model(
                params, self.data[0], smooth=True, data_name=self.data[0]["name"]
            )

        if self.marg:
            mask = self.data[0]["m_w_mask"]
            mod_fit, mod_fit_odd = mod[mask], mod_odd[mask]
            smooth_fit, smooth_fit_odd = smooth[mask], smooth_odd[mask]

            len_poly = len(self.data[0]["ks"]) if self.isotropic else len(self.data[0]["ks"]) * len(self.data[0]["fit_poles"])
            polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
            polysmooth_fit, polysmooth_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
            for n in range(np.shape(polymod)[0]):
                polymod_fit[n], polymod_fit_odd[n] = polymod[n, mask], polymod_odd[n, mask]
                polysmooth_fit[n], polysmooth_fit_odd[n] = polysmooth[n, mask], polysmooth_odd[n, mask]

            bband = self.get_ML_nuisance(
                self.data[0]["pk"], mod_fit, mod_fit_odd, polymod_fit, polymod_fit_odd, self.data[0]["icov"], self.data[0]["icov_m_w"]
            )
            mod += mod_odd + bband @ (polymod + polymod_odd)
            mod_fit += mod_fit_odd + bband @ (polymod_fit + polymod_fit_odd)

            print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
            print(
                self.get_chi2_likelihood(
                    self.data[0]["pk"], mod_fit, np.zeros(mod_fit.shape), self.data[0]["icov"], self.data[0]["icov_m_w"]
                ),
                len(self.data[0]["ks"]) * len(self.data[0]["fit_poles"]),
            )
            print(self.get_alphas(params["alpha"], params["epsilon"]))

            bband = self.get_ML_nuisance(
                self.data[0]["pk"],
                smooth_fit,
                smooth_fit_odd,
                polysmooth_fit,
                polysmooth_fit_odd,
                self.data[0]["icov"],
                self.data[0]["icov_m_w"],
            )
            smooth += smooth_odd + bband @ (polysmooth + polysmooth_odd)
        else:
            mod += mod_odd
            smooth += smooth_odd

        # Mask the model to match the data points
        mod = mod[self.data[0]["w_mask"]]
        smooth = smooth[self.data[0]["w_mask"]]

        # Split up the different multipoles if we have them
        if len(err) > len(ks):
            assert len(err) % len(ks) == 0, f"Cannot split your data - have {len(err)} points and {len(ks)} modes"
        errs = [row for row in err.reshape((-1, len(ks)))]
        mods = [row for row in mod.reshape((-1, len(ks)))]
        smooths = [row for row in smooth.reshape((-1, len(ks)))]
        if self.isotropic:
            names = [f"pk0"]
        else:
            names = [f"pk{n}" for n in self.data[0]["poles"]]
        labels = [f"$P_{{{n}}}(k)$" for n in self.data[0]["poles"]]
        num_rows = len(names)
        cs = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        height = 2 + 1.4 * num_rows

        fig, axes = plt.subplots(figsize=(9, height), nrows=num_rows, ncols=2, sharex=True, squeeze=False)
        ratio = (height - 1) / height
        plt.subplots_adjust(left=0.1, top=ratio, bottom=0.05, right=0.85, hspace=0, wspace=0.3)
        for ax, err, mod, smooth, name, label, c in zip(axes, errs, mods, smooths, names, labels, cs):

            # Plot ye old data
            ax[0].errorbar(ks, ks * self.data[0][name], yerr=ks * err, fmt="o", ms=4, label="Data", c=c)
            ax[1].errorbar(ks, ks * (self.data[0][name] - smooth), yerr=ks * err, fmt="o", ms=4, label="Data", c=c)

            # Plot ye old model
            ax[0].plot(ks, ks * mod, c=c, label="Model")
            ax[1].plot(ks, ks * (mod - smooth), c=c, label="Model")

            if name in [f"pk{n}" for n in self.data[0]["poles"] if n % 2 == 0]:
                ax[0].set_ylabel("$k \\times $ " + label)
            else:
                ax[0].set_ylabel("$ik \\times $ " + label)

            if name not in [f"pk{n}" for n in self.data[0]["fit_poles"]]:
                ax[0].set_facecolor("#e1e1e1")
                ax[1].set_facecolor("#e1e1e1")

        # Show the model parameters
        self.data[0]["icov_m_w"] = icov_m_w
        string = f"$\\mathcal{{L}}$: {self.get_likelihood(params, self.data[0]):0.3g}\n"
        if self.marg:
            string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items() if v not in self.fix_params])
        else:
            string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items()])
        va = "center" if self.postprocess is None else "top"
        ypos = 0.5 if self.postprocess is None else 0.98
        fig.text(0.99, ypos, string, horizontalalignment="right", verticalalignment=va)
        axes[-1, 0].set_xlabel("k")
        axes[-1, 1].set_xlabel("k")
        axes[0, 0].legend(frameon=False)

        if self.postprocess is None:
            axes[0, 1].set_title("$P(k) - P_{\\rm smooth}(k)$")
        else:
            axes[0, 1].set_title("$P(k) - data$")
        axes[0, 0].set_title("$k \\times P(k)$")

        fig.suptitle(self.data[0]["name"] + " + " + self.get_name())
        if figname is not None:
            fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)
        plt.show()


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_power_Beutler2017.py")
    print("bao_power_Ding2018.py")
    print("bao_power_Noda2019.py")
    print("bao_power_Seo2016.py")
