from functools import lru_cache

from scipy.integrate import simps
from scipy.interpolate import splev, splrep
from scipy.linalg import block_diag

from barry.cosmology.power_spectrum_smoothing import smooth_func, validate_smooth_method
from barry.models.model import Model, Omega_m_z, Correction
import numpy as np

from barry.utils import break_vector_and_get_blocks


class PowerSpectrumFit(Model):
    """Generic power spectrum model"""

    def __init__(
        self,
        name="Pk Basic",
        smooth_type=None,
        fix_params=("om",),
        postprocess=None,
        smooth=False,
        recon=None,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        n_poly=0,
        n_data=1,
        data_share_bias=False,
        data_share_poly=False,
    ):
        """Generic power spectrum function model

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
        super().__init__(name, postprocess=postprocess, correction=correction, isotropic=isotropic, marg=marg, n_data=n_data)
        if smooth_type is None:
            smooth_type = {"method": "hinton2017"}
        self.smooth_type = smooth_type
        if not validate_smooth_method(self.smooth_type):
            exit(0)

        self.n_data_bias = 1 if data_share_bias else self.n_data
        self.n_data_poly = 1 if data_share_poly else self.n_data

        if n_poly > 0 and self.isotropic:
            poly_poles = [0]

        self.n_poly = n_poly
        self.poly_poles = poly_poles
        self.data_share_poly = data_share_poly

        self.recon = False
        self.recon_type = "None"
        if recon is not None:
            if recon.lower() != "None":
                if recon.lower() == "iso":
                    self.recon_type = "iso"
                elif recon.lower() == "ani":
                    self.recon_type = "ani"
                elif recon.lower() == "sym":
                    self.recon_type = "sym"
                else:
                    raise ValueError("recon not recognised, must be 'iso', 'ani' or 'sym'")
                self.recon = True

        self.declare_parameters()

        # Set up data structures for model fitting
        self.smooth = smooth

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

        self.kvals = None
        self.pksmooth = None
        self.pkratio = None

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

    def set_data(self, data, parent=False):
        """Sets the models data, including fetching the right cosmology and PT generator.

        Note that if you pass in multiple datas (ie a list with more than one element),
        they need to have the same cosmology.

        Parameters
        ----------
        data : dict, list[dict]
            A list of datas to use
        """
        super().set_data(data)
        if not parent:
            self.set_bias(data[0])

    def set_bias(self, data, kval=0.2, width=0.4):
        """Sets the bias default value by comparing the data monopole and linear pk

        Parameters
        ----------
        data : dict
            The data to use
        kval: float
            The value of k at which to perform the comparison. Default 0.2

        """

        kmax = np.amax(data["ks"])
        if kval > kmax:
            kval = np.amax(data["ks"])
            self.logger.info(f"Default kval for setting beta prior (0.2) larger than kmax={kmax:4.3f}, setting kval=kmax")

        c = data["cosmology"]
        for i in range(self.n_data_bias):
            datapk = splev(kval, splrep(data["ks"], data["pk0"][i]))
            cambpk = self.camb.get_data(om=c["om"], h0=c["h0"])
            modelpk = splev(kval, splrep(cambpk["ks"], cambpk["pk_lin"]))
            kaiserfac = datapk / modelpk
            f = self.get_default("f") if self.param_dict.get("f") is not None else Omega_m_z(c["om"], c["z"]) ** 0.55
            b = -1.0 / 3.0 * f + np.sqrt(kaiserfac - 4.0 / 45.0 * f**2)
            if not self.marg:
                min_b, max_b = (1.0 - width) * b, (1.0 + width) * b
                self.set_default(f"b{{{0}}}_{{{i+1}}}", b**2, min_b**2, max_b**2)
                self.logger.info(f"Setting default bias to b{{{0}}}_{{{i+1}}}={b:0.5f} with {width:0.5f} fractional width")
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
        for i in range(self.n_data_bias):
            self.add_param(f"b{{{0}}}_{{{i+1}}}", f"$b{{{0}}}_{{{i+1}}}$", 0.1, 10.0, 1.0)  # Galaxy bias
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch for monopole
        if not self.isotropic:
            self.add_param("epsilon", r"$\epsilon$", -0.2, 0.2, 0.0)  # Stretch for multipoles

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """Computes the smoothed linear power spectrum and the wiggle ratio

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

        pk_smooth_lin = smooth_func(
            self.camb.ks,
            res["pk_lin"],
            om=om,
            h0=self.camb.h0,
            ob=self.camb.omega_b,
            ns=self.camb.ns,
            rs=res["r_s"],
            **self.smooth_type,
        )  # Get the smoothed power spectrum
        pk_ratio = res["pk_lin"] / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

    @lru_cache(maxsize=32)
    def get_kprimefac(self, epsilon):
        """Computes the prefactor to dilate a k value given epsilon, such that kprime = k * kprimefac / alpha

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
        kprimefac = np.sqrt(musq / epsilonsq**2 + (1.0 - musq) * epsilonsq)
        return kprimefac

    @lru_cache(maxsize=32)
    def get_muprime(self, epsilon):
        """Computes dilated values of mu given input values of epsilon for the power spectrum

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
        muprime = self.mu / np.sqrt(musq + (1.0 + epsilon) ** 6 * (1.0 - musq))
        return muprime

    def integrate_mu(self, pk2d, isotropic=False):
        pk0 = simps(pk2d, self.mu, axis=1)
        if isotropic:
            pk2 = None
            pk4 = None
        else:
            pk2 = 3.0 * simps(pk2d * self.mu**2, self.mu)
            pk4 = 1.125 * (35.0 * simps(pk2d * self.mu**4, self.mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0)
        return pk0, pk2, pk4

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None):
        """Get raw ks and p(k) multipoles for a given parametrisation dilated based on the values of alpha and epsilon

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
        # Get the basic power spectrum components
        if self.kvals is None or self.pksmooth is None or self.pkratio is None:
            ks = self.camb.ks
            pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])
        else:
            ks = self.kvals
            pk_smooth_lin, pk_ratio = self.pksmooth, self.pkratio

        # We split for isotropic and anisotropic here for consistency with our previous isotropic convention, which
        # differs from our implementation of the Beutler2017 isotropic model quite a bit. This results in some duplication
        # of code and a few nested if statements, but it's perhaps more readable and a little faster (because we only
        # need one interpolation for the whole isotropic monopole, rather than separately for the smooth and wiggle components)

        if self.isotropic:
            pk = [np.zeros(len(k))]
            kprime = k if for_corr else k / p["alpha"]
            pk_smooth = splev(kprime, splrep(ks, pk_smooth_lin))
            if not for_corr:
                pk_smooth *= p["b{0}"]

            if smooth:
                pk[0] = pk_smooth if for_corr else pk_smooth
            else:
                # Compute the propagator
                C = np.ones(len(ks))
                propagator = splev(kprime, splrep(ks, (1.0 + pk_ratio * C)))
                pk[0] = pk_smooth * propagator

            poly = np.zeros((1, len(k)))
            if self.marg:
                prefac = np.ones(len(kprime)) if smooth else propagator
                poly = prefac * [pk_smooth]

        else:

            epsilon = 0 if for_corr else p["epsilon"]
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.mu if for_corr else self.get_muprime(epsilon)
            if self.recon_type.lower() == "iso":
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - splev(kprime, splrep(self.camb.ks, self.camb.smoothing_kernel)))
            else:
                kaiser_prefac = 1.0 + p["beta"] * muprime**2
            pk_smooth = kaiser_prefac**2 * splev(kprime, splrep(ks, pk_smooth_lin))
            if not for_corr:
                pk_smooth *= p["b{0}"]

            # Compute the propagator
            if smooth:
                pk2d = pk_smooth
            else:
                C = np.ones(len(ks))
                pk2d = pk_smooth * (1.0 + splev(kprime, splrep(ks, pk_ratio)) * C)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

            # Polynomial shape
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]

            if for_corr:
                poly = None
                kprime = k
            else:
                if self.marg:
                    poly = np.zeros((1, 6, len(k)))
                    poly[0, :, :] = pk
                    pk = [np.zeros(len(k))] * 6
                else:
                    poly = np.zeros((1, 6, len(k)))

        return kprime, pk, poly

    def add_poly(self, k, kpoly, p, prefac, pk):
        """Returns the polynomial components for 3 terms per multipole

        Parameters
        ----------
        k : np.ndarray
            Array of k values for the shape terms
        kpoly : np.ndarray
            Array of k values for the marginalised polynomial array
        p : dict
            dictionary of parameter name to float value pairs
        prefac : np.ndarray
            Prefactors to be added to the front of the analytically marginalised polynomial
        pk_comp: np.ndarray
            the power spectrum components without polynomials

        Returns
        -------
        shape : np.ndarray
            The polynomial terms to be added directly to each multipole
        poly: np.ndarray
            The additive terms in the model not added to the multipoles, necessary for analytical marginalisation

        """

        if self.isotropic:
            shape = np.zeros(len(k))
            for ip in range(self.n_poly):
                shape += p[f"a{{{0}}}_{{{ip+1}}}"] * k ** (ip - 1)

            if self.marg:
                poly = np.zeros((self.n_poly + 1, 1, len(kpoly)))
                poly[0, :, :] = prefac * pk
                for ip in range(self.n_poly):
                    poly[ip + 1, :, :] = kpoly ** (ip - 1)
            else:
                poly = np.zeros((1, 1, len(kpoly)))
        else:
            shape = np.zeros((6, len(k)))
            if self.marg:
                poly = np.zeros((self.n_poly * len(self.poly_poles) + 1, 6, len(kpoly)))
                poly[0, :, :] = pk
                polyvec = [kpoly ** (ip - 1) for ip in range(self.n_poly)]
                for i, pole in enumerate(self.poly_poles):
                    poly[self.n_poly * i + 1 : self.n_poly * (i + 1) + 1, pole] = polyvec

            else:
                poly = np.zeros((1, 6, len(kpoly)))
                for pole in self.poly_poles:
                    for ip in range(self.n_poly):
                        shape[pole] += p[f"a{{{pole}}}_{{{ip+1}}}"] * k ** (ip - 1)

        return shape, poly

    def adjust_model_window_effects(self, pk_generated, data, window=True, wide_angle=True):
        """Take the window effects into account.

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
            # to make the window function correction more similar between this and anisotropic.
            if window:
                p0 = np.sum(data["w_scale"] * pk_generated)
                integral_constraint = data["w_pk"] * p0

                pk_convolved = data["w_transform"] @ pk_generated
                pk_normalised = (pk_convolved - integral_constraint).flatten()
            else:
                pk_normalised = []
                pkin = np.split(pk_generated, data["ndata"])
                for i in range(data["ndata"]):
                    pk_normalised.append(splev(data["ks_output"], splrep(data["ks_input"], pkin[i])))
                pk_normalised = np.hstack(pk_normalised)

        else:

            if window:
                if wide_angle:
                    # Convolve the model and apply wide-angle effects to compute odd multipoles
                    pk_normalised = data["w_m_transform"] @ pk_generated
                else:
                    # Only convolve the model, but don't compute odd multipoles
                    pk_normalised = data["w_transform"] @ pk_generated
            else:
                if wide_angle:
                    # Compute odd multipoles, but no window function convolution
                    pk_generated = data["m_transform"] @ pk_generated
                # Interpolate to the correct k
                pk_normalised = []
                nk = len(data["poles"]) * len(data["ks_input"])
                for i in range(data["ndata"]):
                    for l in range(len(data["poles"][data["poles"]])):
                        pk_normalised.append(
                            splev(
                                data["ks_output"],
                                splrep(
                                    data["ks_input"],
                                    pk_generated[i * nk + l * len(data["ks_input"]) : i * nk + (l + 1) * len(data["ks_input"])],
                                ),
                            )
                        )
                pk_normalised = np.array(pk_normalised).flatten()

        return pk_normalised

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
        num_data = len(d["pk"])

        pk_model, pk_model_odd, poly_model, poly_model_odd, mask = self.get_model(p, d, smooth=self.smooth, data_name=d["name"])

        if self.isotropic or d["icov_m_w"][0] is None:
            pk_model, pk_model_odd = pk_model[mask], pk_model_odd[mask]

        if self.marg:
            if self.isotropic or d["icov_m_w"][0] is None:
                len_poly = d["ndata"] * len(d["ks"]) if self.isotropic else d["ndata"] * len(d["ks"]) * len(d["fit_poles"])
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
                num_data=num_data,
            )
        elif self.marg_type == "full":
            return self.get_chi2_marg_likelihood(
                d["pk"],
                pk_model,
                pk_model_odd,
                poly_model_fit,
                poly_model_fit_odd,
                d["icov"],
                d["icov_m_w"],
                num_mocks=num_mocks,
                num_data=num_data,
            )
        else:
            return self.get_chi2_likelihood(
                d["pk"],
                pk_model,
                pk_model_odd,
                d["icov"],
                d["icov_m_w"],
                num_mocks=num_mocks,
                num_data=num_data,
            )

    def deal_with_ndata(self, params, i):

        p = params.copy()
        p["b{0}"] = params[f"b{{{0}}}_{{{1}}}"] if self.n_data_bias == 1 else params[f"b{{{0}}}_{{{i+1}}}"]
        for pole in self.poly_poles:
            for ip in range(self.n_poly):
                p[f"a{{{pole}}}_{{{ip+1}}}"] = (
                    p[f"a{{{pole}}}_{{{ip+1}}}_{{{1}}}"] if self.n_data_poly == 1 else params[f"a{{{pole}}}_{{{ip+1}}}_{{{i+1}}}"]
                )

        return p

    def get_model(self, params, d, smooth=False, data_name=None, window=True):
        """Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        params : dict
            A dictionary of parameter names to parameter values
        d : dict
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

        # Loop over the constituent (correlated) datasets in d and generate their models
        all_pks = []
        all_polys = []
        for i in range(d["ndata"]):
            p = self.deal_with_ndata(params, i)

            # Generate the underlying models
            ks, pks, poly = self.compute_power_spectrum(d["ks_input"], p, smooth=smooth, data_name=data_name)
            all_pks.append(pks)
            all_polys.append(poly)

        all_polys = np.array(all_polys)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        # Split the model into even and odd components
        if self.isotropic:
            pk_generated = np.concatenate([all_pks[i][0] for i in range(d["ndata"])])
        else:
            pk_generated = np.concatenate(
                [np.concatenate([all_pks[i][l] for i in range(d["ndata"])]) for l in d["poles"][d["poles"] % 2 == 0]]
            )

        if self.isotropic:
            pk_model, mask = self.adjust_model_window_effects(pk_generated, d, window=window), d["m_w_mask"]
            if self.postprocess is not None:
                pk_model = self.postprocess(ks=d["ks_output"], pk=pk_model, mask=mask)
            pk_model_odd = np.zeros(d["ndata"] * len(d["ks_output"]))
        else:
            pk_model_odd = np.zeros(d["ndata"] * 6 * len(ks))
            pk_model_odd[1 * d["ndata"] * len(ks) : 2 * d["ndata"] * len(ks)] += np.concatenate([all_pks[i][1] for i in range(d["ndata"])])
            pk_model_odd[3 * d["ndata"] * len(ks) : 4 * d["ndata"] * len(ks)] += np.concatenate([all_pks[i][3] for i in range(d["ndata"])])
            pk_model_odd[5 * d["ndata"] * len(ks) : 6 * d["ndata"] * len(ks)] += np.concatenate([all_pks[i][5] for i in range(d["ndata"])])
            if d["icov_m_w"][0] is None:
                pk_model, mask = self.adjust_model_window_effects(pk_generated, d, window=window), d["m_w_mask"]
                pk_model_odd = self.adjust_model_window_effects(pk_model_odd, d, window=window, wide_angle=False)
            else:
                pk_model, mask = pk_generated, np.ones(len(pk_generated), dtype=bool)

        poly_model, poly_model_odd = None, None
        if self.marg:

            # Concatenate the poly matrix in the correct way for our datasets based on the parameters they are sharing
            nmarg, nell = np.shape(all_polys)[1:3]
            if self.n_data_bias == 1:
                if self.n_data_poly == 1:
                    # Everything shared
                    poly = np.array([[np.concatenate(all_polys[:, n, l]) for l in range(nell)] for n in range(nmarg)])
                else:
                    # Only bias shared
                    poly_b = np.array([[np.concatenate(all_polys[:, 0, l]) for l in range(nell)]]) if nmarg > self.n_poly else []
                    poly_poly = np.array([block_diag(*all_polys[:, 1:, l]) for l in range(nell)]).transpose((1, 0, 2))
                    poly = np.vstack([poly_b, poly_poly])
            else:
                if self.n_data_poly == 1:
                    # Only polynomials shared
                    poly_b = (
                        np.array([block_diag(*all_polys[:, 0, l]) for l in range(nell)]).transpose((1, 0, 2)) if nmarg > self.n_poly else []
                    )
                    poly_poly = np.array([[np.concatenate(all_polys[:, n + 1, l]) for l in range(nell)] for n in range(self.n_poly)])
                    poly = np.vstack([poly_b, poly_poly])
                else:
                    # Nothing shared
                    poly = np.array([block_diag(*all_polys[:, :, l]) for l in range(nell)]).transpose((1, 0, 2))

            nell = np.shape(poly)[1] - 1 if d["icov_m_w"][0] is None else np.shape(poly)[1] - 3
            len_poly_k = d["ndata"] * len(d["ks_output"]) if d["icov_m_w"][0] is None else d["ndata"] * len(d["ks_input"])
            len_poly = d["ndata"] * len(d["ks_output"]) if self.isotropic else len_poly_k * nell
            len_poly_odd = d["ndata"] * len(d["ks_output"]) if self.isotropic else len_poly_k * (np.shape(poly)[1] - 1)
            poly_model = np.zeros((np.shape(poly)[0], len_poly))
            poly_model_odd = np.zeros((np.shape(poly)[0], len_poly_odd))
            for n in range(np.shape(poly)[0]):

                # poly_split = np.split(poly[n], d["ndata"], axis=-1)

                if self.isotropic:
                    poly_generated = poly[n, 0]
                else:
                    poly_generated = np.concatenate([poly[n, l, :] for l in d["poles"][d["poles"] % 2 == 0]])

                if self.isotropic:
                    poly_model_long = self.adjust_model_window_effects(poly_generated, d, window=window)
                    if self.postprocess is not None:
                        poly_model_long = self.postprocess(ks=d["ks_output"], pk=poly_model_long, mask=mask)
                    poly_model[n] = poly_model_long
                else:
                    nk = d["ndata"] * len(ks)
                    poly_generated_odd = np.zeros(d["ndata"] * 6 * len(ks))
                    poly_generated_odd[1 * nk : 2 * nk] += poly[n, 1]
                    poly_generated_odd[3 * nk : 4 * nk] += poly[n, 3]
                    poly_generated_odd[5 * nk : 6 * nk] += poly[n, 5]
                    if d["icov_m_w"][0] is None:
                        poly_model[n] = self.adjust_model_window_effects(poly_generated, d, window=window)
                        poly_model_odd[n] = self.adjust_model_window_effects(poly_generated_odd, d, window=window, wide_angle=False)
                    else:
                        poly_model[n], poly_model_odd[n] = poly_generated, poly_generated_odd

        return pk_model, pk_model_odd, poly_model, poly_model_odd, mask

    def get_model_summary(self, params, window=True, smooth_params=None):
        """Get the model summary for the given parameters.

        Parameters
        ----------
        params : array
            The parameter vector.
        smooth_params : array, optional
            The parameter vector for the smooth model.

        Returns
        -------

        """
        # Ensures we return the window convolved model
        icov_m_w = self.data[0]["icov_m_w"]
        self.data[0]["icov_m_w"][0] = None

        mod, mod_odd, polymod, polymod_odd, _ = self.get_model(params, self.data[0], data_name=self.data[0]["name"], window=window)
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

            len_poly = (
                self.data[0]["ndata"] * len(self.data[0]["ks"])
                if self.isotropic
                else self.data[0]["ndata"] * len(self.data[0]["ks"]) * len(self.data[0]["fit_poles"])
            )
            polymod_fit, polymod_fit_odd = np.empty((np.shape(polymod)[0], len_poly)), np.zeros((np.shape(polymod)[0], len_poly))
            polysmooth_fit, polysmooth_fit_odd = np.empty((np.shape(polysmooth)[0], len_poly)), np.zeros(
                (np.shape(polysmooth)[0], len_poly)
            )
            for n in range(np.shape(polymod)[0]):
                polymod_fit[n], polymod_fit_odd[n] = polymod[n, mask], polymod_odd[n, mask]
                polysmooth_fit[n], polysmooth_fit_odd[n] = polysmooth[n, mask], polysmooth_odd[n, mask]

            bband = self.get_ML_nuisance(
                self.data[0]["pk"], mod_fit, mod_fit_odd, polymod_fit, polymod_fit_odd, self.data[0]["icov"], self.data[0]["icov_m_w"]
            )
            mod = mod + mod_odd + bband @ (polymod + polymod_odd)
            mod_fit = mod_fit + mod_fit_odd + bband @ (polymod_fit + polymod_fit_odd)

            # print(f"Maximum likelihood nuisance parameters at maximum a posteriori point are {bband}")
            new_chi_squared = -2.0 * self.get_chi2_likelihood(
                self.data[0]["pk"],
                mod_fit,
                np.zeros(mod_fit.shape),
                self.data[0]["icov"],
                self.data[0]["icov_m_w"],
                num_mocks=self.data[0]["num_mocks"],
                num_data=len(self.data[0]["pk"]),
            )
            alphas = params["alpha"] if self.isotropic else self.get_alphas(params["alpha"], params["epsilon"])
            dof = len(self.data[0]["pk"]) - len(self.get_active_params()) - len(bband)

            bband_smooth = self.get_ML_nuisance(
                self.data[0]["pk"],
                smooth_fit,
                smooth_fit_odd,
                polysmooth_fit,
                polysmooth_fit_odd,
                self.data[0]["icov"],
                self.data[0]["icov_m_w"],
            )
            smooth = smooth + smooth_odd + bband_smooth @ (polysmooth + polysmooth_odd)
        else:
            mod = mod + mod_odd
            smooth = smooth + smooth_odd
            dof = len(self.data[0]["pk"]) - len(self.get_active_params())
            new_chi_squared = 0.0
            bband = None

        # Mask the model to match the data points
        mod = mod[self.data[0]["w_mask"]]
        smooth = smooth[self.data[0]["w_mask"]]

        mods = mod.reshape((-1, self.data[0]["ndata"], len(self.data[0]["ks"])))
        smooths = smooth.reshape((-1, self.data[0]["ndata"], len(self.data[0]["ks"])))

        self.data[0]["icov_m_w"] = icov_m_w

        return new_chi_squared, dof, bband, mods, smooths

    def plot(self, params, window=True, smooth_params=None, figname=None, title=None, display=True):
        import matplotlib.pyplot as plt

        ks = self.data[0]["ks"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))

        new_chi_squared, dof, bband, mods, smooths = self.get_model_summary(params, window=window, smooth_params=smooth_params)

        # Split up the different multipoles if we have them
        if len(err) > len(ks):
            assert len(err) % len(ks) == 0, f"Cannot split your data - have {len(err)} points and {len(ks)} modes"
        errs = err.reshape((-1, self.data[0]["ndata"], len(ks)))
        if self.isotropic:
            names = [f"pk0"]
        else:
            names = [f"pk{n}" for n in self.data[0]["poles"]]
        labels = [f"$P_{{{n}}}(k)$" for n in self.data[0]["poles"]]
        num_rows = len(names)
        cs = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        height = 2 + 1.4 * num_rows

        if display is True or figname is not None:

            fig, axes = plt.subplots(figsize=(9, height), nrows=num_rows, ncols=2, sharex=True, squeeze=False)
            ratio = (height - 1) / height
            plt.subplots_adjust(left=0.1, top=ratio, bottom=0.05, right=0.85, hspace=0, wspace=0.3)
            for ax, err, mod, smooth, name, label, c in zip(axes, errs, mods, smooths, names, labels, cs):

                # print(name, err, mod, smooth)

                for i in range(self.data[0]["ndata"]):

                    mfc = c if i == 0 else "w"
                    ls = "-" if i == 0 else "--"
                    l = "Data" if i == 0 else None

                    # Plot ye old data
                    ax[0].errorbar(ks, ks * self.data[0][name][i], yerr=ks * err[i], fmt="o", ms=4, label=l, mec=c, mfc=mfc, c=c)
                    ax[1].errorbar(
                        ks,
                        ks * (self.data[0][name][i] - smooth[i]),
                        yerr=ks * err[i],
                        fmt="o",
                        ms=4,
                        label=l,
                        mec=c,
                        mfc=mfc,
                        c=c,
                    )

                    l = "Model" if i == 0 else None

                    # Plot ye old model
                    ax[0].plot(ks, ks * mod[i], c=c, ls=ls, label=l)
                    ax[0].plot(ks, ks * smooth[i], c=c, ls="--", label=l)
                    ax[1].plot(ks, ks * (mod[i] - smooth[i]), c=c, ls=ls, label=l)

                    if name in [f"pk{n}" for n in self.data[0]["poles"] if n % 2 == 0]:
                        ax[0].set_ylabel("$k \\times $ " + label)
                    else:
                        ax[0].set_ylabel("$ik \\times $ " + label)

                    if name not in [f"pk{n}" for n in self.data[0]["fit_poles"]]:
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
            axes[-1, 0].set_xlabel("k")
            axes[-1, 1].set_xlabel("k")
            axes[0, 0].legend(frameon=False)

            if self.postprocess is None:
                axes[0, 1].set_title("$P(k) - P_{\\rm smooth}(k)$")
            else:
                axes[0, 1].set_title("$P(k) - data$")
            axes[0, 0].set_title("$k \\times P(k)$")

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
    print("bao_power_Beutler2017.py")
    print("bao_power_Ding2018.py")
    print("bao_power_Noda2019.py")
    print("bao_power_Seo2016.py")
