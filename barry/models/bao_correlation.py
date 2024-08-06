from functools import lru_cache
import numpy as np

from barry.cosmology.pk2xi import PowerToCorrelationSphericalBessel
from barry.cosmology.power_spectrum_smoothing import validate_smooth_method
from barry.models.model import Model, Omega_m_z
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep
from scipy.integrate import simpson
from scipy.special import sici

from barry.utils import break_vector_and_get_blocks


class CorrelationFunctionFit(Model):
    """A generic model for computing correlation functions."""

    def __init__(
        self,
        name="Corr Basic",
        smooth_type=None,
        recon=None,
        fix_params=("om",),
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        includeb2=False,
        include_binmat=True,
        broadband_type="spline",
        **kwargs,
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
            broadband_type=None,
        )
        if not validate_smooth_method(self.parent.smooth_type):
            exit(0)

        self.n_data_bias = 1
        self.n_data_poly = 1

        self.broadband_type = None if broadband_type is None else broadband_type.lower()
        if self.broadband_type not in [None, "poly", "spline"]:
            raise ValueError("broadband_type not recognised, must be None, 'poly' or 'spline'")

        self.n_poly = [] if self.broadband_type is None else kwargs.get("n_poly", [0, 2])
        if self.broadband_type == "spline":
            self.delta_fac = kwargs.get("delta", 2.0)
            self.logger.info(f"Including low order splines in broadband with delta={self.delta_fac}*pi/r_s")

        if (len(self.n_poly) > 0 or self.broadband_type == "spline") and self.isotropic:
            poly_poles = [0]

        self.recon = recon
        self.poly_poles = poly_poles
        self.data_share_poly = True

        nspline = 2 if 2 in self.poly_poles and self.broadband_type == "spline" else 0
        self.len_poly = len(self.n_poly) * len(self.poly_poles) + nspline

        self.includeb2 = includeb2
        self.declare_parameters()

        self.include_binmat = include_binmat

        # Set up data structures for model fitting
        self.smooth = smooth

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.pk2xi_0 = None
        self.pk2xi_2 = None
        self.pk2xi_4 = None

        self.poly = None  # Basic polynomial terms
        self.winpoly = None  # Polynomial terms convolved with the window function

    def set_marg(self, fix_params, do_bias=False, marg_bias=0):

        self.marg_bias = marg_bias

        if self.marg:
            fix_params = list(fix_params)
            if do_bias:
                for i in range(self.n_data_bias):
                    fix_params.extend([f"b{{{0}}}_{{{i+1}}}"])
                if self.includeb2:
                    for pole in self.poly_poles:
                        if pole != 0:
                            fix_params.extend([f"b{{{pole}}}_{{{i+1}}}"])
            for i in range(self.n_data_poly):
                if self.broadband_type == "spline" and 2 in self.poly_poles:
                    fix_params.extend([f"bbspline_{{{0}}}_{{{1}}}"])
                    fix_params.extend([f"bbspline_{{{1}}}_{{{1}}}"])
                for pole in self.poly_poles:
                    for ip in self.n_poly:
                        fix_params.extend([f"a{{{pole}}}_{{{ip}}}_{{{i+1}}}"])

        self.set_fix_params(fix_params)

        if self.marg:
            if do_bias:
                for i in range(self.n_data_bias):
                    self.set_default(f"b{{{0}}}_{{{i+1}}}", 1.0)
                    if self.includeb2:
                        for pole in self.poly_poles:
                            if pole != 0:
                                self.set_default(f"b{{{pole}}}_{{{i+1}}}", 1.0)
            for i in range(self.n_data_poly):
                if self.broadband_type == "spline" and 2 in self.poly_poles:
                    self.set_default(f"bbspline_{{{0}}}_{{{1}}}", 0.0)
                    self.set_default(f"bbspline_{{{1}}}_{{{1}}}", 0.0)
                for pole in self.poly_poles:
                    for ip in self.n_poly:
                        self.set_default(f"a{{{pole}}}_{{{ip}}}_{{{i+1}}}", 0.0)

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
        cambpk = self.camb.get_data(om=data[0]["cosmology"]["om"], h0=data[0]["cosmology"]["h0"])
        self.pk2xi_0 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=0)
        self.pk2xi_2 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=2)
        self.pk2xi_4 = PowerToCorrelationSphericalBessel(qs=cambpk["ks"], ell=4)
        self.set_bias(data[0])
        self.parent.set_data(data, parent=True)
        if self.broadband_type == "spline":
            self.delta = self.delta_fac * np.pi / self.camb.get_data(om=data[0]["cosmology"]["om"], h0=data[0]["cosmology"]["h0"])["r_s"]
            self.logger.info(f"Broadband Delta fixed to {self.delta}")

    def set_bias(self, data, sval=50.0, width=0.5):
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
        modelxi = self.pk2xi_0.__call__(cambpk["ks"], cambpk["pk_lin_z"], np.array([sval]))[0]
        kaiserfac = dataxi / modelxi
        f = self.param_dict.get("f") if self.param_dict.get("f") is not None else Omega_m_z(c["om"], c["z"]) ** 0.55
        b = -1.0 / 3.0 * f + np.sqrt(kaiserfac - 4.0 / 45.0 * f**2) if kaiserfac - 4.0 / 45.0 * f**2 > 0 else 1.0
        if not self.marg_bias:
            if self.get_default(f"b{{{0}}}_{{{1}}}") is None:
                min_b, max_b = (1.0 - width) * b, (1.0 + width) * b
                self.set_default(f"b{{{0}}}_{{{1}}}", b**2, min=min_b**2, max=max_b**2)
                self.logger.info(f"Setting default bias to b0={b:0.5f} with {width:0.5f} fractional width")
                if self.includeb2:
                    for pole in self.poly_poles:
                        self.set_default(f"b{{{pole}}}_{{{1}}}", b**2, min=min_b**2, max=max_b**2)
                        self.logger.info(f"Setting default bias to b{{{pole}}}={b:0.5f} with {width:0.5f} fractional width")
            else:
                self.logger.info(f"Using default bias parameter of b0={self.get_default(f'b{{{0}}}_{{{1}}}'):0.5f}")
        if self.param_dict.get("beta") is not None:
            if self.get_default("beta") is None:
                beta, beta_min, beta_max = f / b, (1.0 - width) * f / b, (1.0 + width) * f / b
                self.set_default("beta", beta, beta_min, beta_max)
                self.logger.info(f"Setting default RSD parameter to beta={beta:0.5f} with {width:0.5f} fractional width")
            else:
                self.logger.info(f"Using default RSD parameter of beta={self.get_default('beta'):0.5f}")

    def declare_parameters(self):
        """Defines model parameters, their bounds and default value."""
        self.add_param(f"b{{{0}}}_{{{1}}}", f"$b_{{{0},{1}}}$", 0.01, 10.0, 1.0)  # Linear galaxy bias for each multipole
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles
        for pole in self.poly_poles:
            for ip in self.n_poly:
                self.add_param(f"a{{{pole}}}_{{{ip}}}_{{{1}}}", f"$a_{{{pole},{ip},1}}$", -10.0, 10.0, 0)
        if self.broadband_type == "spline" and 2 in self.poly_poles:
            self.add_param(f"bbspline_{{{0}}}_{{{1}}}", f"$bbspline_{{0,1}}$", -10.0, 10.0, 0)
            self.add_param(f"bbspline_{{{1}}}_{{{1}}}", f"$bbspline_{{1,1}}$", -10.0, 10.0, 0)

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
        xi0 = simpson(xi2d, x=self.mu, axis=1)
        if isotropic:
            xi2 = None
            xi4 = None
        else:
            xi2 = 3.0 * simpson(xi2d * mu**2, x=self.mu, axis=1)
            xi4 = 35.0 * simpson(xi2d * mu**4, x=self.mu, axis=1)
        return xi0, xi2, xi4

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
            and 3 bias parameters and polynomial terms per multipole

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

        _, pks = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth)
        xi_comp = np.array([self.pk2xi_0.__call__(self.parent.camb.ks, pks[0], dist), np.zeros(len(dist)), np.zeros(len(dist))])

        if not self.isotropic:
            xi_comp[1] = self.pk2xi_2.__call__(self.parent.camb.ks, pks[2], dist)
            xi_comp[2] = self.pk2xi_4.__call__(self.parent.camb.ks, pks[4], dist)

        return dist, xi_comp

    def compute_poly(self, dist):
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
        # Prefactor, roughly equal to the typical k_min/2pi, set so that the free coefficients are all of the same order
        A = 0.02 / (2.0 * np.pi)

        if self.isotropic:
            self.poly = np.zeros((self.len_poly, 1, len(dist)))
            for i, ip in enumerate(self.n_poly):
                self.poly[i, :, :] = (A * dist) ** ip
        else:
            nspline = 2 if self.broadband_type == "spline" else 0
            self.poly = np.zeros((self.len_poly, 3, len(dist)))
            polyvec = [(A * dist) ** ip for ip in self.n_poly]
            if len(self.n_poly) > 0:
                for i, pole in enumerate(self.poly_poles):
                    self.poly[len(self.n_poly) * i : len(self.n_poly) * (i + 1), int(pole / 2)] = polyvec

            # Add on the spline broadband terms (B2m1 and B20) for the quadrupole if requested.
            # TODO: Should also compute and code up relevant B4 terms in the event we fit the hexadecapole?
            if self.broadband_type == "spline":
                x = self.delta * dist
                sinx, sin2x, sin3x = np.sin(x), np.sin(2.0 * x), np.sin(3.0 * x)
                cosx, cos2x, cos3x = np.cos(x), np.cos(2.0 * x), np.cos(3.0 * x)
                Si_x, Si_2x, Si_3x = sici(x)[0], sici(2.0 * x)[0], sici(3.0 * x)[0]
                self.poly[-2, 1] = self.delta**3 * (
                    -2.0
                    * (
                        12.0
                        - 16.0 * cosx
                        + x**2 * cosx
                        + 4.0 * cos2x
                        - x**2 * cos2x
                        - x * sinx
                        + x * cosx * sinx
                        + x**3 * Si_x
                        - 2.0 * x**3 * Si_2x
                    )
                    / x**6
                )
                self.poly[-1, 1] = self.delta**3 * (
                    0.5
                    * (
                        48.0
                        + 8.0 * x**2
                        - 96.0 * cosx
                        + 6.0 * x**2 * cosx
                        + 64.0 * cos2x
                        - 16.0 * x**2 * cos2x
                        - 16.0 * cos3x
                        + 9.0 * x**2 * cos3x
                        - 6.0 * x * sinx
                        + 8.0 * x * sin2x
                        - 3.0 * x * sin3x
                        + 6.0 * x**3 * Si_x
                        - 32.0 * x**3 * Si_2x
                        + 27.0 * x**3 * Si_3x
                    )
                    / x**3
                )

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

        dist, xis = self.compute_correlation_function(d["dist_input"] if self.include_binmat else d["dist"], p, smooth=smooth)

        # If we are not analytically marginalising, we now add the polynomial terms on to the correlation function
        if not self.marg:
            if self.poly is None:
                self.compute_poly(d["dist_input"] if self.include_binmat else d["dist"])
            for pole in self.poly_poles:
                for i, ip in enumerate(self.n_poly):
                    xis[int(pole / 2)] += p[f"a{{{pole}}}_{{{ip}}}_{{{1}}}"] * self.poly[i, int(pole / 2), :]
            if self.broadband_type == "spline":
                xis[1] += p[f"bbspline_{{{0}}}_{{{1}}}"] * self.poly[-2, 1, :]
                xis[1] += p[f"bbspline_{{{1}}}_{{{1}}}"] * self.poly[-1, 1, :]

        # Convolve the xi model with the binning matrix and concatenate into a single data vector
        xi_generated = [xi @ d["binmat"] if self.include_binmat else xi for xi in xis]
        if self.isotropic:
            xi_model = xi_generated[0]
        else:
            xi_model = np.concatenate([xi_generated[l] for l in range(len(d["poles"]))])

        # Now sort out the polynomial terms in the case of analytical marginalisation
        poly_model = None
        if self.marg:

            # Load the polynomial terms if computed already, or compute them for the first time.
            if self.winpoly is None:
                if self.poly is None:
                    self.compute_poly(d["dist_input"] if self.include_binmat else d["dist"])

                nmarg, nell = np.shape(self.poly)[:2]
                len_poly = len(d["dist"]) if self.isotropic else len(d["dist"]) * nell
                self.winpoly = np.zeros((nmarg, len_poly))
                for n in range(nmarg):
                    self.winpoly[n] = np.concatenate([pol @ d["binmat"] if self.include_binmat else pol for pol in self.poly[n]])

            if self.marg_bias > 0:
                # We need to update/include the poly terms corresponding to the galaxy bias
                poly_model = np.vstack([xi_model, self.winpoly])
                xi_model = np.zeros(len(xi_model))
            else:
                poly_model = self.winpoly

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
                poly_model_fit,
                d["icov"],
                num_mocks=num_mocks,
                num_data=num_data,
            )
        elif self.marg_type == "full":
            return self.get_chi2_marg_likelihood(
                d["xi"],
                xi_model_fit,
                poly_model_fit,
                d["icov"],
                num_mocks=num_mocks,
                num_data=num_data,
                marg_bias=self.marg_bias,
            )
        else:
            return self.get_chi2_likelihood(
                d["xi"],
                xi_model_fit,
                d["icov"],
                num_mocks=num_mocks,
                num_data=num_data,
            )

    def get_model_summary(self, params, smooth_params=None, verbose=False):

        ss = self.data[0]["dist"]
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
            bband = self.get_ML_nuisance(self.data[0]["xi"], mod_fit, polymod_fit, self.data[0]["icov"])
            mod = mod + bband @ polymod
            mod_fit = mod_fit + bband @ polymod_fit

            if verbose:
                print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
            new_chi_squared = -2.0 * self.get_chi2_likelihood(
                self.data[0]["xi"],
                mod_fit,
                self.data[0]["icov"],
                num_mocks=self.data[0]["num_mocks"],
                num_data=len(self.data[0]["xi"]),
            )
            dof = len(self.data[0]["xi"]) - len(self.get_active_params()) - len(bband)
            if verbose:
                print(f"Chi squared/dof is {new_chi_squared}/{dof} at these values")

            bband_smooth = self.get_ML_nuisance(
                self.data[0]["xi"],
                smooth_fit,
                polysmooth_fit,
                self.data[0]["icov"],
            )
            smooth = smooth + bband @ polysmooth
        else:
            dof = len(self.data[0]["xi"]) - len(self.get_active_params())
            new_chi_squared = 0.0
            bband = None

        mods = [row for row in mod.reshape((-1, len(ss)))]
        smooths = [row for row in smooth.reshape((-1, len(ss)))]

        return new_chi_squared, dof, bband, mods, smooths

    def plot(self, params, smooth_params=None, figname=None, title=None, display=True):
        import matplotlib.pyplot as plt

        # Ensures we plot the window convolved model
        ss = self.data[0]["dist"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))

        new_chi_squared, dof, bband, mods, smooths = self.get_model_summary(params, smooth_params=smooth_params)

        # Split up the different multipoles if we have them
        if len(err) > len(ss):
            assert len(err) % len(ss) == 0, f"Cannot split your data - have {len(err)} points and {len(ss)} bins"
        errs = [row for row in err.reshape((-1, len(ss)))]
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
                ax[0].plot(ss, ss**2 * smooth, c=c, ls="--", label="Smooth")
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

    def simple_plot(self, params, smooth_params=None, figname=None, title=None, display=True, c="r"):
        import matplotlib.pyplot as plt

        ss = self.data[0]["dist"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))

        new_chi_squared, dof, bband, mods, smooths = self.get_model_summary(params, smooth_params)

        # Split up the different multipoles if we have them
        if len(err) > len(ss):
            assert len(err) % len(ss) == 0, f"Cannot split your data - have {len(err)} points and {len(ss)} bins"
        errs = [row for row in err.reshape((-1, len(ss)))]
        if self.isotropic:
            names = [f"xi0"]
        else:
            names = [f"xi{n}" for n in self.data[0]["fit_poles"]]
        num_rows = len(names)
        ms = ["o", "o", "s"]
        mfcs = [c, "w", c]
        height = 2 + 1.4 * num_rows

        if display is True or figname is not None:
            self.logger.info("Create plot")

            fig, axes = plt.subplots(figsize=(height, height), nrows=1, ncols=1, sharex=True, squeeze=False)
            for err, mod, smooth, name, m, mfc in zip(errs, mods, smooths, names, ms, mfcs):

                # Plot ye old data
                axes[0, 0].errorbar(ss, ss**2 * self.data[0][name], yerr=ss**2 * err, fmt=m, mfc=mfc, label="Data", c=c)

                # Plot ye old model
                axes[0, 0].plot(ss, ss**2 * mod, c=c, label="Model")
                axes[0, 0].plot(ss, ss**2 * smooth, c=c, ls="--", label="Smooth")

            axes[0, 0].set_xlabel("$s\,(h^{-1}\,\mathrm{Mpc})$")
            axes[0, 0].set_ylabel("$s^{2} \\times \\xi(s)$ ")

            # Add the chi_squared and dof
            string = f"$\\chi^{2}/$dof$=${new_chi_squared:.1f}$/${dof:d}\n"
            axes[0, 0].text(0.02, 0.98, string, horizontalalignment="left", verticalalignment="top", transform=axes[0, 0].transAxes)
            axes[0, 0].text(
                0.98,
                0.98,
                f"$\\alpha$=${params['alpha']:.4f}$\n",
                horizontalalignment="right",
                verticalalignment="top",
                transform=axes[0, 0].transAxes,
            )
            if not self.isotropic:
                axes[0, 0].text(
                    0.98,
                    0.94,
                    f"$\\alpha_{{{{ap}}}}$=${(1.0 + params['epsilon']) ** 3:.4f}$\n",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=axes[0, 0].transAxes,
                )

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
