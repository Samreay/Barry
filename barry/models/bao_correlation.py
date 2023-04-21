from functools import lru_cache
import numpy as np

from barry.cosmology.pk2xi import PowerToCorrelationFFTLog, PowerToCorrelationGauss, PowerToCorrelationSphericalBessel
from barry.cosmology.power_spectrum_smoothing import validate_smooth_method, smooth_func
from barry.models.model import Model, Omega_m_z, Correction
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep
from scipy.integrate import simps

from barry.utils import break_vector_and_get_blocks


class CorrelationFunctionFit(Model):
    """A generic model for computing correlation functions."""

    def __init__(
        self,
        name="Corr Basic",
        smooth_type=None,
        recon=None,
        fix_params=("om"),
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        includeb2=True,
        n_poly=3,
    ):

        """Generic correlation function model

        Parameters
        ----------
        name : str, optional
            Name of the model
            Name of the mode
        smooth_type : str, optional
            The sort of smoothing to use. Either 'hinton2017' or 'eh1998'
        fix_params : list[str], optional
            Parameter names to fix to their defaults. Defaults to just `[om]`.
        smooth : bool, optional
            Whether to generate a smooth model without the BAO feature. Defaults to `false`.
        correction : `Correction` enum. Defaults to `Correction.SELLENTIN
        """
        super().__init__(name, correction=correction, isotropic=isotropic, marg=marg)
        self.parent = PowerSpectrumFit(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            n_poly=n_poly,
        )
        if smooth_type is None:
            self.smooth_type = {"method": "hinton2017"}
        if not validate_smooth_method(self.smooth_type):
            exit(0)

        self.n_data_bias = 1
        self.n_data_poly = 1

        if n_poly > 0 and self.isotropic:
            poly_poles = [0]

        self.recon = recon
        self.n_poly = n_poly
        self.poly_poles = poly_poles
        self.data_share_poly = True

        self.declare_parameters()
        self.includeb2 = includeb2

        # Set up data structures for model fitting
        self.smooth = smooth

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.pk2xi_0 = None
        self.pk2xi_2 = None
        self.pk2xi_4 = None

        self.fixed_xi = False
        self.store_xi = [None, None, None]
        self.store_xi_smooth = [None, None, None]

    def set_marg(self, fix_params, poly_poles, n_poly, do_bias=False):

        if self.marg:
            fix_params = list(fix_params)
            if do_bias:
                for i in range(self.n_data_bias):
                    fix_params.extend([f"b{{{0}}}_{{{i+1}}}"])
            for i in range(self.n_data_poly):
                for pole in poly_poles:
                    for ip in range(n_poly):
                        fix_params.extend([f"a{{{pole}}}_{{{ip+1}}}_{{{i+1}}}"])

        self.set_fix_params(fix_params)

        if self.marg:
            if do_bias:
                for i in range(self.n_data_bias):
                    self.set_default(f"b{{{0}}}_{{{i+1}}}", 1.0)
            for i in range(self.n_data_poly):
                for pole in self.poly_poles:
                    for ip in range(n_poly):
                        self.set_default(f"a{{{pole}}}_{{{ip+1}}}_{{{i+1}}}", 0.0)

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
        # self.pk2xi_0 = PowerToCorrelationGauss(self.camb.ks, ell=0, interpolateDetail=20)
        # self.pk2xi_2 = PowerToCorrelationGauss(self.camb.ks, ell=2, interpolateDetail=20)
        # self.pk2xi_4 = PowerToCorrelationGauss(self.camb.ks, ell=4, interpolateDetail=20)
        # self.pk2xi_0 = PowerToCorrelationFFTLog(ell=0, r0=50.0)
        # self.pk2xi_2 = PowerToCorrelationFFTLog(ell=2, r0=50.0)
        # self.pk2xi_4 = PowerToCorrelationFFTLog(ell=4, r0=50.0)
        cambpk = self.camb.get_data(om=data[0]["cosmology"]["om"], h0=data[0]["cosmology"]["h0"])
        self.pk2xi_0 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=0)
        self.pk2xi_2 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=2)
        self.pk2xi_4 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=4)
        self.set_bias(data[0])
        self.parent.set_data(data, parent=True)

    def set_bias(self, data, sval=50.0, width=0.3):
        """Sets the bias default value by comparing the data monopole and linear model

        Parameters
        ----------
        data : dict
            The data to use
        kval: float
            The value of k at which to perform the comparison. Default 0.2

        """

        c = data["cosmology"]
        dataxi = splev(sval, splrep(data["dist"], data["xi0"]))
        cambpk = self.camb.get_data(om=c["om"], h0=c["h0"])
        modelxi = self.pk2xi_0.__call__(cambpk["ks"], cambpk["pk_lin"], np.array([sval]))[0]
        kaiserfac = dataxi / modelxi
        f = self.param_dict.get("f") if self.param_dict.get("f") is not None else Omega_m_z(c["om"], c["z"]) ** 0.55
        b = -1.0 / 3.0 * f + np.sqrt(kaiserfac - 4.0 / 45.0 * f**2)
        if not self.marg:
            min_b, max_b = (1.0 - width) * b, (1.0 + width) * b
            # if self.param_dict.get("b") is not None:
            #    self.set_default(f"b", b**2, min=min_b**2, max=max_b**2)
            #    self.logger.info(f"Setting default bias to b={b:0.5f} with {width:0.5f} fractional width")
            # else:
            self.set_default(f"b{{{0}}}_{{{1}}}", b**2, min=min_b**2, max=max_b**2)
            self.logger.info(f"Setting default bias to b0={b:0.5f} with {width:0.5f} fractional width")
        if self.param_dict.get("beta") is not None:
            if self.get_default("beta") is None:
                beta, beta_min, beta_max = f / b, (1.0 - width) * f / b, (1.0 + width) * f / b
                self.set_default("beta", beta, beta_min, beta_max)
                self.logger.info(f"Setting default RSD parameter to beta={beta:0.5f} with {width:0.5f} fractional width")
            else:
                beta = self.get_default("beta")
                self.logger.info(f"Using default RSD parameter of beta={beta:0.5f}")

    def declare_parameters(self):
        """Defines model parameters, their bounds and default value."""
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles

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
        musq = self.mu**2
        epsilonsq = (1.0 + epsilon) ** 2
        sprimefac = np.sqrt(musq * epsilonsq**2 + (1.0 - musq) / epsilonsq)
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
        musq = self.mu**2
        muprime = self.mu / np.sqrt(musq + (1.0 - musq) / (1.0 + epsilon) ** 6)
        return muprime

    def integrate_mu(self, xi2d, mu=None, isotropic=False):
        if mu is None:
            mu = self.mu
        xi0 = simps(xi2d, self.mu, axis=1)
        if isotropic:
            xi2 = None
            xi4 = None
        else:
            xi2 = 3.0 * simps(xi2d * mu**2, self.mu, axis=1)
            xi4 = 35.0 * simps(xi2d * mu**4, self.mu, axis=1)
        return xi0, xi2, xi4

    def compute_basic_correlation_function(self, dist, p, smooth=False):
        """Computes the basic correlation function computes usig the parent Power spectrum class

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
        xi : np.ndarray
            the model monopole, quadrupole and hexadecapole components interpolated to sprime.
        """

        if self.fixed_xi:
            if smooth:
                if self.store_xi_smooth[0] is None:
                    ks, pks, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)
            else:
                if self.store_xi[0] is None:
                    ks, pks, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)
        else:
            ks, pks, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)

        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        finedist = np.linspace(0.0, 300.0, 601)
        if self.isotropic:
            sprime = p["alpha"] * dist
            if self.fixed_xi:
                if smooth:
                    if self.store_xi_smooth[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        self.store_xi_smooth[0] = pk2xi0
                    else:
                        pk2xi0 = self.store_xi_smooth[0]
                else:
                    if self.store_xi[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        self.store_xi[0] = pk2xi0
                    else:
                        pk2xi0 = self.store_xi[0]
            else:
                pk2xi0 = self.pk2xi_0.__call__(ks, pks[0], sprime)
            xi[0] = splev(sprime, pk2xi0)
        else:
            # Construct the dilated 2D correlation function by splining the undilated multipoles. We could have computed these
            # directly at sprime, but sprime depends on both s and mu, so splining is quicker
            # epsilon = np.round(p["epsilon"], decimals=5)
            epsilon = p["epsilon"]
            sprime = np.outer(dist * p["alpha"], self.get_sprimefac(epsilon))
            muprime = self.get_muprime(epsilon)

            if self.fixed_xi:
                if smooth:
                    if self.store_xi_smooth[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                        pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))
                        self.store_xi_smooth = [pk2xi0, pk2xi2, pk2xi4]
                    else:
                        pk2xi0, pk2xi2, pk2xi4 = self.store_xi_smooth
                else:
                    if self.store_xi[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                        pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))
                        self.store_xi = [pk2xi0, pk2xi2, pk2xi4]
                    else:
                        pk2xi0, pk2xi2, pk2xi4 = self.store_xi
            else:
                pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))

            xi0 = splev(sprime, pk2xi0)
            xi2 = splev(sprime, pk2xi2)
            xi4 = splev(sprime, pk2xi4)

            xi2d = xi0 + 0.5 * (3.0 * muprime**2 - 1) * xi2 + 0.125 * (35.0 * muprime**4 - 30.0 * muprime**2 + 3.0) * xi4

            # Now compute the dilated xi multipoles
            xi[0], xi[1], xi[2] = self.integrate_mu(xi2d, self.mu)

        return sprime, xi

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
            and 3 bias parameters but no polynomial terms

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
        xi : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """
        sprime, xi_comp = self.compute_basic_correlation_function(dist, p, smooth=smooth)
        xi, poly = self.add_poly(dist, p, xi_comp)

        return sprime, xi, poly

    def add_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but with 3 polynomial terms for each multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = xi0
            poly = np.zeros((1, len(dist)))
            if self.marg:
                poly = [1.0 / dist ** (ip - 1) for ip in range(self.n_poly)]
            else:
                for ip in range(self.n_poly):
                    xi[0] += p[f"a{0}_{{{ip+1}}}_{1}"] / (dist ** (ip - 1))

        else:
            xi = [xi0, xi2, xi4]

            # Polynomial shape
            if self.marg:
                poly = np.zeros((self.n_poly * len(self.poly_poles), 3, len(dist)))
                for npole, pole in enumerate(self.poly_poles):
                    poly[self.n_poly * npole : self.n_poly * (npole + 1), npole] = [1.0 / dist ** (ip - 1) for ip in range(self.n_poly)]
            else:
                poly = np.zeros((1, 3, len(dist)))
                for pole in self.poly_poles:
                    for ip in range(self.n_poly):
                        xi[int(pole / 2)] += p[f"a{{{pole}}}_{{{ip+1}}}_{1}"] / dist ** (ip - 1)

        return xi, poly

    def add_three_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but with 3 polynomial terms for each multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = xi0
            poly = np.zeros((1, len(dist)))
            if self.marg:
                poly = [1.0 / (dist**2), 1.0 / dist, np.ones(len(dist))]
            else:
                xi[0] += p["a{0}_1_{1}"] / (dist**2) + p["a{0}_2_{1}"] / dist + p["a{0}_3_{1}"]

        else:
            xi = [xi0, xi2, xi4]

            # Polynomial shape
            if self.marg:
                poly = np.zeros((3 * len(self.poly_poles), 3, len(dist)))
                for npole, pole in enumerate(self.poly_poles):
                    poly[3 * npole : 3 * (npole + 1), npole] = [1.0 / (dist**2), 1.0 / dist, np.ones(len(dist))]
            else:
                poly = np.zeros((1, 3, len(dist)))
                for pole in self.poly_poles:
                    xi[int(pole / 2)] += (
                        p[f"a{{{pole}}}_1_{{{1}}}"] / dist**2 + p[f"a{{{pole}}}_2_{{{1}}}"] / dist + p[f"a{{{pole}}}_3_{{{1}}}"]
                    )

        return xi, poly

    def add_zero_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but with 3 polynomial terms for each multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = xi0
            poly = np.zeros((1, len(dist)))
        else:
            xi = [xi0, xi2, xi4]
            poly = np.zeros((3 * len(self.poly_poles), 3, len(dist)))

        return xi, poly

    def get_model(self, p, d, smooth=False):
        """Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.
        smooth : bool, optional
            Whether to only generate a smooth model without the BAO feature

        Returns
        -------
        xi_model : np.ndarray
            The concatenated xi_{\\ell}(s) predictions at the dilated distances given p and data['dist']
        poly_model : np.ndarray
            the functions describing any polynomial terms, used for analytical marginalisation
            k values correspond to d['dist']
        """

        dist, xis, poly = self.compute_correlation_function(d["dist_input"], p, smooth=smooth)

        # Convolve the xi model with the binning matrix
        xi_generated = [xi @ d["binmat"] for xi in xis]

        xi_model = xi_generated[0] if self.isotropic else np.concatenate([xi_generated[0], xi_generated[1]])
        if 4 in d["poles"] and not self.isotropic:
            xi_model = np.concatenate([xi_model, xi_generated[2]])

        poly_model = None
        if self.marg:
            len_poly = len(d["dist"])
            if not self.isotropic:
                len_poly *= len(d["poles"])
            poly_model = np.empty((np.shape(poly)[0], len_poly))
            for n in range(np.shape(poly)[0]):
                if self.isotropic:
                    poly_model[n] = poly[n] @ d["binmat"]
                else:
                    poly_generated = [pol @ d["binmat"] for pol in poly[n]]
                    if 4 in d["poles"]:
                        poly_model[n] = np.concatenate(poly_generated)
                    else:
                        poly_model[n] = np.concatenate([poly_generated[0], poly_generated[1]])

        return xi_model, poly_model

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
        num_mocks = d["num_mocks"]
        num_data = len(d["xi"])

        xi_model, poly_model = self.get_model(p, d, smooth=self.smooth)

        if self.isotropic:
            xi_model_fit = xi_model
            poly_model_fit = poly_model
        else:
            xi_model_fit = break_vector_and_get_blocks(xi_model, len(d["poles"]), d["fit_pole_indices"])
            if self.marg:
                poly_model_fit = np.empty((np.shape(poly_model)[0], len(self.data[0]["fit_pole_indices"]) * len(self.data[0]["dist"])))
                for n in range(np.shape(poly_model)[0]):
                    poly_model_fit[n] = break_vector_and_get_blocks(
                        poly_model[n], np.shape(poly_model)[1] / len(d["dist"]), d["fit_pole_indices"]
                    )

        if self.marg_type == "partial":
            return self.get_chi2_partial_marg_likelihood(
                d["xi"],
                xi_model_fit,
                np.zeros(xi_model_fit.shape),
                poly_model_fit,
                np.zeros(poly_model_fit.shape),
                d["icov"],
                [None],
                num_mocks=num_mocks,
                num_data=num_data,
            )
        elif self.marg_type == "full":
            return self.get_chi2_marg_likelihood(
                d["xi"],
                xi_model_fit,
                np.zeros(xi_model_fit.shape),
                poly_model_fit,
                np.zeros(poly_model_fit.shape),
                d["icov"],
                [None],
                num_mocks=num_mocks,
                num_data=num_data,
            )
        else:
            return self.get_chi2_likelihood(
                d["xi"],
                xi_model_fit,
                np.zeros(xi_model_fit.shape),
                d["icov"],
                [None],
                num_mocks=num_mocks,
                num_data=num_data,
            )

    def plot(self, params, smooth_params=None, figname=None, title=None, display=True):
        import matplotlib.pyplot as plt

        # Ensures we plot the window convolved model
        ss = self.data[0]["dist"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        mod, polymod = self.get_model(params, self.data[0])
        if smooth_params is not None:
            smooth, polysmooth = self.get_model(smooth_params, self.data[0], smooth=True)
        else:
            smooth, polysmooth = self.get_model(params, self.data[0], smooth=True)

        if self.marg:
            if self.isotropic:
                mod_fit = mod
                smooth_fit = smooth
                polymod_fit = polymod
                polysmooth_fit = polysmooth
            else:
                mod_fit = break_vector_and_get_blocks(mod, len(self.data[0]["poles"]), self.data[0]["fit_pole_indices"])
                smooth_fit = break_vector_and_get_blocks(smooth, len(self.data[0]["poles"]), self.data[0]["fit_pole_indices"])
                polymod_fit = np.empty((np.shape(polymod)[0], len(self.data[0]["fit_pole_indices"]) * len(self.data[0]["dist"])))
                polysmooth_fit = np.empty((np.shape(polysmooth)[0], len(self.data[0]["fit_pole_indices"]) * len(self.data[0]["dist"])))
                for n in range(np.shape(polymod)[0]):
                    polymod_fit[n] = break_vector_and_get_blocks(
                        polymod[n], np.shape(polymod)[1] / len(self.data[0]["dist"]), self.data[0]["fit_pole_indices"]
                    )
                    polysmooth_fit[n] = break_vector_and_get_blocks(
                        polysmooth[n], np.shape(polysmooth)[1] / len(self.data[0]["dist"]), self.data[0]["fit_pole_indices"]
                    )
            bband = self.get_ML_nuisance(
                self.data[0]["xi"], mod_fit, np.zeros(mod_fit.shape), polymod_fit, np.zeros(polymod_fit.shape), self.data[0]["icov"], [None]
            )
            mod = mod + bband @ polymod
            mod_fit = mod_fit + bband @ polymod_fit

            # print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
            new_chi_squared = -2.0 * self.get_chi2_likelihood(
                self.data[0]["xi"],
                mod_fit,
                np.zeros(mod_fit.shape),
                self.data[0]["icov"],
                [None],
                num_mocks=self.data[0]["num_mocks"],
                num_data=len(self.data[0]["xi"]),
            )
            alphas = params["alpha"] if self.isotropic else self.get_alphas(params["alpha"], params["epsilon"])
            dof = len(self.data[0]["xi"]) - len(self.get_active_params()) - len(bband)

            bband_smooth = self.get_ML_nuisance(
                self.data[0]["xi"],
                smooth_fit,
                np.zeros(smooth_fit.shape),
                polysmooth_fit,
                np.zeros(polysmooth_fit.shape),
                self.data[0]["icov"],
                [None],
            )
            smooth = smooth + bband_smooth @ polysmooth
        else:
            dof = len(self.data[0]["xi"]) - len(self.get_active_params())
            new_chi_squared = 0.0
            bband = None

        # Split up the different multipoles if we have them
        if len(err) > len(ss):
            assert len(err) % len(ss) == 0, f"Cannot split your data - have {len(err)} points and {len(ss)} bins"
        errs = [row for row in err.reshape((-1, len(ss)))]
        mods = [row for row in mod.reshape((-1, len(ss)))]
        smooths = [row for row in smooth.reshape((-1, len(ss)))]
        if self.isotropic:
            names = [f"xi0"]
        else:
            names = [f"xi{n}" for n in self.data[0]["poles"]]
        labels = [f"$\\xi_{{{n}}}(s)$" for n in self.data[0]["poles"]]
        num_rows = len(names)
        cs = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        height = 2 + 1.4 * num_rows

        if display is True or figname is not None:
            self.logger.info("Create plot")

            fig, axes = plt.subplots(figsize=(9, height), nrows=num_rows, ncols=2, sharex=True, squeeze=False)
            ratio = (height - 1) / height
            plt.subplots_adjust(left=0.1, top=ratio, bottom=0.05, right=0.85, hspace=0, wspace=0.3)
            for ax, err, mod, smooth, name, label, c in zip(axes, errs, mods, smooths, names, labels, cs):

                # Plot ye old data
                ax[0].errorbar(ss, ss**2 * self.data[0][name], yerr=ss**2 * err, fmt="o", ms=4, label="Data", c=c)
                ax[1].errorbar(ss, ss**2 * (self.data[0][name] - smooth), yerr=ss**2 * err, fmt="o", ms=4, label="Data", c=c)

                # Plot ye old model
                ax[0].plot(ss, ss**2 * mod, c=c, label="Model")
                ax[1].plot(ss, ss**2 * (mod - smooth), c=c, label="Model")

                ax[0].set_ylabel("$s^{2} \\times $ " + label)

                if name not in [f"xi{n}" for n in self.data[0]["fit_poles"]]:
                    ax[0].set_facecolor("#e1e1e1")
                    ax[1].set_facecolor("#e1e1e1")

            # Show the model parameters
            string = f"$\\mathcal{{L}}$: {self.get_likelihood(params, self.data[0]):0.3g}\n"
            if self.marg:
                string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items() if l not in self.fix_params])
            else:
                string += "\n".join([f"{self.param_dict[l].label}={v:0.4g}" for l, v in params.items()])
            va = "center" if self.postprocess is None else "top"
            ypos = 0.5 if self.postprocess is None else 0.98
            fig.text(0.99, ypos, string, horizontalalignment="right", verticalalignment=va)
            axes[-1, 0].set_xlabel("s")
            axes[-1, 1].set_xlabel("s")
            axes[0, 0].legend(frameon=False)

            if self.postprocess is None:
                axes[0, 1].set_title("$\\xi(s) - \\xi_{\\rm smooth}(s)$")
            else:
                axes[0, 1].set_title("$\\xi(s) - data$")
            axes[0, 0].set_title("$s^{2} \\times \\xi(s)$")

            if title is None:
                title = self.data[0]["name"] + " + " + self.get_name()
            fig.suptitle(title)
            if figname is not None:
                fig.savefig(figname, bbox_inches="tight", transparent=True, dpi=300)
            if display:
                plt.show()

        return new_chi_squared, dof, bband, mods, smooths


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_correlation_Ross2017.py")
    print("bao_correlation_Ding2018.py")
    print("bao_correlation_Seo2016.py")
