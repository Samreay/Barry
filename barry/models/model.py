import os
import inspect
import logging
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from numpy.random import uniform, normal
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import truncnorm
from scipy.special import loggamma
from scipy.optimize import basinhopping, differential_evolution
from enum import Enum, unique
from dataclasses import dataclass
from functools import lru_cache


from barry.cosmology.camb_generator import Omega_m_z, getCambGenerator


@dataclass
class Param:
    name: str
    label: str
    min: float
    max: float
    sigma: float
    default: float
    prior: str
    active: bool


@unique
class Correction(Enum):
    """Various corrections that we should apply when computing our likelihood.

    NONE gives no correction.
    HARTLAP implements the chi2 correction given in Hartlap 2007
    SELLENTIN implements the correction from Sellentin 2016

    """

    NONE = 0
    HARTLAP = 1
    SELLENTIN = 2


class Model(ABC):
    """Abstract model class.

    Implement a version of this that overwrites the `plot` and get_likelihood` methods.

    """

    def __init__(self, name, postprocess=None, correction=None, isotropic=False, marg=None, n_data=1):
        """Create a new model.

        Parameters
        ----------
        name : str
            The name of the model
        postprocess : `Postprocess` class, optional
            The postprocessing class to apply to the model before computing the likelihood. This is not applied automatically.
        correction : `Correction` class, optional
            Correction to apply to the likelihood. Applied automatically. Defaults to `Correction.SELLENTIN`
        isotropic: bool, optional
            Whether or not the model is isotropic. Defaults to True
        """
        self.name = name
        self.logger = logging.getLogger("barry")
        self.data = None
        self.data_dict = None
        self.n_data = n_data

        # For pregeneration
        self.camb = None
        self.cosmology = None
        self.isotropic = isotropic
        self.pregen = None
        self.pregen_path = None
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../generated/")
        os.makedirs(self.data_location, exist_ok=True)

        self.params = []
        self.fix_params = []
        self.param_dict = {}
        self.postprocess = postprocess
        if postprocess is not None and not self.isotropic:
            raise NotImplementedError("Postprocessing (i.e., BAOExtractor) not implemented for anisotropic fits")
        if correction is None:
            correction = Correction.NONE
        self.correction = correction
        self.correction_data = {}  # Empty dict to store correction specific data for speeding up computation
        assert isinstance(self.correction, Correction), "Correction should be an enum of Correction"
        self.logger.info(
            f"Created model {name} of {self.__class__.__name__} with correction {correction} and postprocess {str(postprocess)}"
        )

        self.marg = False
        self.marg_type = "None"
        if marg is not None:
            self.marg_type = "full"
            if marg.lower() == "partial":
                self.marg_type = "partial"
            self.marg = True
            self.logger.info(f"Using {self.marg_type} analytic marginalisation")
            assert (
                self.correction != Correction.SELLENTIN
            ), "ERROR: SELLENTIN covariance matrix correction not compatible with analytic marginalisation. Switch to Correction.HARTLAP or use marg=false"

    def get_name(self):
        return self.name

    def get_unique_cosmo_name(self):
        """Unique name used to save out any pregenerated data."""
        return self.__class__.__name__ + "_" + self.camb.filename_unique + ".pkl"

    def set_cosmology(self, c, load_pregen=True):
        z = c["z"]
        if self.param_dict.get("f") is not None:
            f = Omega_m_z(self.get_default("om"), z) ** 0.55
            self.set_default("f", f)
            self.logger.info(f"Setting default growth rate of structure to f={f:0.5f}")

        if self.cosmology != c:
            mnu = c.get("mnu", 0.0)
            c["mnu"] = mnu
            self.set_default("om", c["om"])
            if "om" in self.fix_params:
                self.camb = getCambGenerator(
                    om_resolution=1,
                    h0=c["h0"],
                    ob=c["ob"],
                    redshift=c["z"],
                    ns=c["ns"],
                    mnu=c["mnu"],
                    recon_smoothing_scale=c["reconsmoothscale"],
                )
                self.camb.omch2s = [(self.get_default("om") - c["ob"]) * c["h0"] ** 2 - c["mnu"] / 93.14]

            else:
                self.camb = getCambGenerator(
                    h0=c["h0"], ob=c["ob"], redshift=c["z"], ns=c["ns"], mnu=c["mnu"], recon_smoothing_scale=c["reconsmoothscale"]
                )
            self.pregen_path = os.path.abspath(os.path.join(self.data_location, self.get_unique_cosmo_name()))
            self.cosmology = c
            if load_pregen:
                self._load_precomputed_data()

    def set_data(self, data):
        """Sets the models data, including fetching the right cosmology and PT generator.

        Note that if you pass in multiple datas (ie a list with more than one element),
        they need to have the same cosmology and must all be isotropic or anisotropic

        Parameters
        ----------
        data : dict, list[dict]
            A list of datas to use
        """
        if not isinstance(data, list):
            data = [data]
        self.data = data
        self.data_dict = dict([(d["name"], d) for d in data])
        self.set_cosmology(data[0]["cosmology"])
        assert data[0]["isotropic"] == self.isotropic, "ERROR: Data and model isotropic mismatch: Data is %s while model is %s" % (
            "isotropic" if data[0]["isotropic"] else "anisotropic",
            "isotropic" if self.isotropic else "anisotropic",
        )

    def overwrite_template(self, ks, pklin, pknw, pk_nonlin_0=None, pk_nonlin_z=None, r_drag=None):
        """Overwrites the camb template power spectrum with the given values. For instance, if you have a fixed template from a file"""

        self.kvals = ks
        self.pksmooth = pknw
        self.pkratio = pklin / pknw - 1.0

        if r_drag is None:
            r_drag = self.camb.get_data()["r_s"]
        if pk_nonlin_0 is None or pk_nonlin_z is None:
            pk_nonlin_0 = self.camb.get_data()["pk_nl_0"]
            pk_nonlin_z = self.camb.get_data()["pk_nl_z"]

        sinterp = splev(ks, splrep(self.camb.ks, self.camb.smoothing_kernel))
        self.pregen = self.precompute(ks=ks, pk_lin=pklin, pk_nonlin_0=pk_nonlin_0, pk_nonlin_z=pk_nonlin_z, r_drag=r_drag, s=sinterp)

    def add_param(self, name, label, min, max, default, sigma=0.0, prior="flat"):
        assert prior.lower() in ["flat", "gaussian"], "ERROR: Prior must be flat or gaussian"
        p = Param(name, label, min, max, sigma, default, prior, name not in self.fix_params)
        self.params.append(p)
        self.param_dict[name] = p

    def set_fix_params(self, params):
        if params is None:
            params = []
        self.fix_params = params
        for p in self.params:
            p.active = p.name not in params

    def get_param(self, name):
        return self.param_dict.get(name, self.get_default(name))

    def get_active_params(self):
        """Returns a list of the active (non-fixed) parameters"""
        return [p for p in self.params if p.active]

    def get_inactive_params(self):
        """Returns a list of the inactive (fixed) parameters"""
        return [p for p in self.params if not p.active]

    def get_default(self, name):
        """Returns the default value of a given parameter name"""
        return self.param_dict[name].default

    def set_default(self, name, default, min=None, max=None, sigma=None, prior="flat"):
        """Sets the default value for a parameter"""
        self.param_dict[name].default = default
        if min is not None:
            self.param_dict[name].min = min
        if max is not None:
            self.param_dict[name].max = max
        if sigma is not None:
            self.param_dict[name].sigma = sigma
        if prior.lower() != "flat":
            assert prior.lower() == "gaussian", "ERROR: Prior must be flat or gaussian"
            self.param_dict[name].prior = prior

    def get_defaults(self):
        """Returns a list of default values for all active parameters"""
        return [x.default for x in self.get_active_params()]

    def get_defaults_dict(self):
        """Returns a list of default values for all active parameters"""
        return {x.name: x.default for x in self.get_active_params()}

    def get_labels(self):
        """Gets a list of the label for all active parameters"""
        return [x.label for x in self.get_active_params()]

    def get_names(self):
        """Get a list of the names for all active parameters"""
        return [x.name for x in self.get_active_params()]

    def get_extents(self):
        """Gets a list of (min, max) extents for all active parameters"""
        return [(x.min, x.max) for x in self.get_active_params()]

    def get_prior(self, params):
        """The prior, checks for flat or truncated gaussian for each parameter.

        Used by the Ensemble and MH samplers, but not by nested sampling methods.

        """
        log_prior = 0.0
        for pname, val in params.items():
            if val < self.param_dict[pname].min or val > self.param_dict[pname].max:
                return -np.inf
            elif self.param_dict[pname].prior == "gaussian":
                log_prior += -0.5 * ((val - self.param_dict[pname].default) / self.param_dict[pname].sigma) ** 2
        return log_prior

    def get_chi2_likelihood(self, data, model, model_odd, icov, icov_m_w, num_mocks=None, num_data=None):
        """Computes the chi2 corrected likelihood.

        Parameters
        ----------
        diff : np.ndarray
            The difference between the model predictions and data observations
        icov : np.ndarray
            Inverted covariance matrix.
        num_mocks : int, optional
            The number of mocks used to estimate the covariance. Used for corrections.
        num_data : int, optional
            The length of the data vector. Used for corrections.

        Returns
        -------
        log_likelihood : float
            The (corrected) log-likelihood value from the computed chi2.
        """

        if icov_m_w[0] is None:
            diff = data - (model + model_odd)
            chi2 = diff.T @ icov @ diff
        else:
            chi2 = (
                data.T @ icov @ data
                - 2.0 * model_odd.T @ icov_m_w[0] @ data
                - 2.0 * model.T @ icov_m_w[1] @ data
                + model_odd.T @ icov_m_w[2] @ model_odd
                + 2.0 * model.T @ icov_m_w[3] @ model_odd
                + model.T @ icov_m_w[4] @ model.T
            )

        if self.correction in [Correction.HARTLAP, Correction.SELLENTIN]:
            assert (
                num_mocks > 0
            ), "Cannot use HARTLAP  or SELLENTIN correction with covariance not determined from mocks. Set correction to Correction.NONE"

        if self.correction is Correction.HARTLAP:  # From Hartlap 2007
            key = f"{num_mocks}_{num_data}"
            if key not in self.correction_data:
                self.correction_data[key] = (num_mocks - num_data - 2.0) / (num_mocks - 1.0)
            c_p = self.correction_data[key]
            return -0.5 * chi2 * c_p
        elif self.correction is Correction.SELLENTIN:  # From Sellentin 2016
            key = f"{num_mocks}_{num_data}"
            if key not in self.correction_data:
                self.correction_data[key] = (
                    loggamma(num_mocks / 2).real
                    - (num_data / 2) * np.log(np.pi * (num_mocks - 1))
                    - loggamma((num_mocks - num_data) * 0.5).real
                )
            c_p = self.correction_data[key]
            log_likelihood = c_p - (num_mocks / 2) * np.log(1 + chi2 / (num_mocks - 1))
            return log_likelihood
        else:
            return -0.5 * chi2

    def get_chi2_marg_likelihood(self, data, model, model_odd, marg_model, marg_model_odd, icov, icov_m_w, num_mocks=None, num_data=None):
        """Computes the chi2 corrected likelihood.

        Parameters
        ----------
        model : np.ndarray
            The model predictions without any nuisance parameters
        marg_model : np.ndarray
            The parts of the model that depend on nuisance parameters
        data : np.ndarray
            The data vector
        icov : np.ndarray
            Inverted covariance matrix.
        num_mocks : int, optional
            The number of mocks used to estimate the covariance. Used for corrections.
        num_params : int, optional
            The number of parameters in the model. Used for corrections.

        Returns
        -------
        log_likelihood : float
            The (corrected) log-likelihood value from the computed chi2.
        """
        if icov_m_w[0] is None:
            model += model_odd
            marg_model += marg_model_odd
            diff = data - model
            F02 = diff @ icov @ diff
            F11 = marg_model @ icov @ diff
            F2 = marg_model @ icov @ marg_model.T
            F2inv = np.linalg.inv(F2)
        else:
            F02 = (
                data @ icov @ data
                - 2.0 * model_odd @ icov_m_w[0] @ data
                - 2.0 * model @ icov_m_w[1] @ data
                + model_odd @ icov_m_w[2] @ model_odd
                + 2.0 * model @ icov_m_w[3] @ model_odd
                + model @ icov_m_w[4] @ model
            )
            F11 = (
                marg_model_odd @ icov_m_w[0] @ data
                + marg_model @ icov_m_w[1] @ data
                - marg_model_odd @ icov_m_w[2] @ model_odd
                - marg_model @ icov_m_w[3] @ model_odd
                - marg_model_odd @ icov_m_w[3].T @ model
                - marg_model @ icov_m_w[4] @ model
            )
            F2 = (
                marg_model_odd @ icov_m_w[2] @ marg_model_odd.T
                + marg_model @ icov_m_w[3] @ marg_model_odd.T
                + marg_model_odd @ icov_m_w[3].T @ marg_model.T
                + marg_model @ icov_m_w[4] @ marg_model.T
            )
            F2inv = np.linalg.inv(F2)
        chi2 = F02 - F11 @ F2inv @ F11

        if self.correction in [Correction.HARTLAP]:
            assert (
                num_mocks > 0
            ), "Cannot use HARTLAP correction with covariance not determined from mocks. Set correction to Correction.NONE"

        if self.correction is Correction.HARTLAP:  # From Hartlap 2007
            key = f"{num_mocks}_{num_data}"
            if key not in self.correction_data:
                self.correction_data[key] = (num_mocks - num_data - 2.0) / (num_mocks - 1.0)
            c_p = self.correction_data[key]
            return -0.5 * (chi2 * c_p + np.log(np.linalg.det(F2)) + np.shape(F2)[0] * np.log(c_p))
        else:
            return -0.5 * (chi2 + np.log(np.linalg.det(F2)))

    def get_chi2_partial_marg_likelihood(
        self, data, model, model_odd, marg_model, marg_model_odd, icov, icov_m_w, num_mocks=None, num_data=None
    ):
        """Computes the chi2 corrected likelihood.

        Parameters
        ----------
        model : np.ndarray
            The model predictions without any nuisance parameters
        marg_model : np.ndarray
            The parts of the model that depend on nuisance parameters
        data : np.ndarray
            The data vector
        icov : np.ndarray
            Inverted covariance matrix.
        num_mocks : int, optional
            The number of mocks used to estimate the covariance. Used for corrections.
        num_params : int, optional
            The number of parameters in the model. Used for corrections.

        Returns
        -------
        log_likelihood : float
            The (corrected) log-likelihood value from the computed chi2.
        """

        # First compute the MLE values for the nuisance parameters
        bband = self.get_ML_nuisance(data, model, model_odd, marg_model, marg_model_odd, icov, icov_m_w)

        # Add these on to the models
        model += bband @ marg_model
        model_odd += bband @ marg_model_odd

        return self.get_chi2_likelihood(data, model, model_odd, icov, icov_m_w, num_mocks=num_mocks, num_data=num_data)

    def get_ML_nuisance(self, data, model, model_odd, marg_model, marg_model_odd, icov, icov_m_w):

        if icov_m_w[0] is None:
            full_model = model + model_odd
            full_marg_model = marg_model + marg_model_odd
            F11 = full_marg_model @ icov @ (data - full_model)
            F2 = full_marg_model @ icov @ full_marg_model.T
            F2inv = np.linalg.inv(F2)
        else:
            F11 = (
                marg_model_odd @ icov_m_w[0] @ data
                + marg_model @ icov_m_w[1] @ data
                - marg_model_odd @ icov_m_w[2] @ model_odd
                - marg_model @ icov_m_w[3] @ model_odd
                - marg_model_odd @ icov_m_w[3].T @ model
                - marg_model @ icov_m_w[4] @ model
            )
            F2 = (
                marg_model_odd @ icov_m_w[2] @ marg_model_odd.T
                + 2.0 * marg_model @ icov_m_w[3] @ marg_model_odd.T
                + marg_model @ icov_m_w[4] @ marg_model.T
            )
            F2inv = np.linalg.inv(F2)

        bband = F2inv @ F11

        return bband

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

    def get_reverse_alphas(self, alpha_par, alpha_perp):
        """Computes values of alpha and epsilon from the input values of alpha_par and alpha_perp

        Parameters
        ----------
        alpha_par : float
            The dilation scale parallel to the line-of-sight
        alpha_perp : float
            The dilation scale perpendicular to the line-of-sight

        Returns
        -------
        alpha : float
            The isotropic dilation scale
        epsilon: float
            The anisotropic warping

        """
        return alpha_perp ** (2.0 / 3.0) * alpha_par ** (1.0 / 3.0), (alpha_par / alpha_perp) ** (1.0 / 3.0) - 1.0

    def get_raw_start(self):
        """Gets starting points for each parameter given prior and min and max constraints"""
        start_random = []
        for x in self.get_active_params():
            if x.prior == "flat":
                start_random.append(uniform(x.min, x.max))
            else:
                start_random.append(truncnorm.rvs(x.min, x.max, loc=x.default, scale=x.sigma))
        return np.array(start_random)

    def _load_precomputed_data(self):
        if self._needs_precompute():
            if self.camb.singleval:
                self.pregen = self.generate_precomputed_data([[0, 0]])[0][2:][0]
            else:
                assert os.path.exists(self.pregen_path), f"You need to pregenerate the required data for {self.pregen_path}"
                with open(self.pregen_path, "rb") as f:
                    self.pregen = pickle.load(f)
                self.logger.info(f"Pregen data loaded from {self.pregen_path}")
        else:
            self.logger.info("Dont need to load any pregen data")

    def _save_precomputed_data(self, data):
        with open(self.pregen_path, "wb") as f:
            pickle.dump(data, f)
            self.logger.info(f"Pregen data saved to {self.pregen_path}")

    def generate_precomputed_data(self, indexes):
        self.logger.info(f"Pregenerating model {self.__class__.__name__} data for {self.camb.filename_unique}")

        data = []
        for i, j in indexes:
            omch2 = self.camb.omch2s[i]
            h0 = self.camb.h0s[j]
            om = omch2 / (h0 * h0) + self.camb.omega_b

            c = self.camb.get_data(om, h0)

            values = self.precompute(
                om=c["om"],
                h0=c["h0"],
                ks=c["ks"],
                pk_lin=c["pk_lin"],
                pk_nonlin_0=c["pk_nl_0"],
                pk_nonlin_z=c["pk_nl_z"],
                r_drag=c["r_s"],
                s=self.camb.smoothing_kernel,
            )

            data.append([i, j, values])
        return data

    def get_pregen(self, key, om, h0=None):
        if h0 is None:
            h0 = self.camb.h0
        data = self.pregen[key]
        if self.camb.singleval:
            return data
        else:
            return self.camb.interpolate(om, h0, data)

    def _needs_precompute(self):
        func2 = getattr(super(type(self), self), self.precompute.__name__)
        return self.precompute.__func__ != func2.__func__

    def precompute(self, om=None, h0=None, ks=None, pk_lin=None, pk_nonlin_0=None, pk_nonlin_z=None, r_drag=None, s=None):
        """A function available for overriding that precomputes values that depend only on the outputs of CAMB.

        Parameters
        ----------
        camb : CambGenerator
            Stores the ks, pklin, etc
        om : float
            Value of Omega_m, which you can use to get the Camb ks, pklin and nonlinear pks
        h0 : float
            Value of h, which you can use as above

        Returns
        -------
        A dictionary mapping specific keywords to their computed values. Or None.
        """
        return None

    def get_start(self, num_walkers=1):
        """Gets an optimised `n` starting points by calculating a best fit starting point using basinhopping"""
        self.logger.info("Getting start position")

        def minimise(scale_params):
            return -self.get_posterior(self.unscale(scale_params))

        close_default = 3
        start_random = self.get_raw_start()
        start_close = [(s + p.default * close_default) / (1 + close_default) for s, p in zip(start_random, self.get_active_params())]

        self.logger.info("Starting basin hopping to find a good starting point")
        res = basinhopping(
            minimise,
            self.scale(start_close),
            niter_success=3,
            niter=30,
            stepsize=0.05,
            minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 600}},
        )

        scaled_start = res.x
        ratio = 0.05  # 5% of the unit hypercube

        mins = np.clip(scaled_start - ratio, 0, 1)
        maxes = np.clip(scaled_start + ratio, 0, 1)

        samples = np.random.uniform(mins, maxes, size=(num_walkers, len(maxes)))

        unscaled_samples = np.array([self.unscale(s) for s in samples])
        self.logger.debug(f"Start samples have shape {unscaled_samples.shape}")

        return unscaled_samples

    def get_num_dim(self):
        """Gets the number of dimensions (active, free parameters) in the model"""
        return len(self.get_active_params())

    def get_start_scaled(self):
        """Gets a scaled (unit hypercube) optimised starting position."""
        return self.scale(self.get_start())

    def get_param_dict(self, params):
        """Converts a list of parameter values into a dictionary of parameter values"""
        ps = OrderedDict([(p.name, v) for p, v in zip(self.get_active_params(), params)])
        ps.update({(p.name, p.default) for p in self.get_inactive_params()})
        return ps

    def get_posterior_scaled(self, scaled):
        """Gets the posterior using an input scaled (unit hypercube) location in parameter space"""
        return self.get_posterior(self.unscale(scaled))

    def get_likelihood(self, params, data):
        """Returns the likelihood given a list of param values and some data. Designed to be overwritten by subclasses"""
        return 0.0

    def get_posterior(self, params):
        """Returns the posterior given a list of param values."""
        ps = self.get_param_dict(params)
        prior = self.get_prior(ps)
        if not np.isfinite(prior):
            return -np.inf
        posterior = prior
        for d in self.data:
            posterior += self.get_likelihood(ps, d)
        return posterior

    def scale(self, params):
        """Scale parameter values to the unit hypercube. If you want other dists and nested sampling, overwrite this"""
        scaled = []
        for s, p in zip(params, self.get_active_params()):
            if p.prior == "flat":
                scaled.append((s - p.min) / (p.max - p.min))
            else:
                scaled.append(truncnorm.cdf(s, (p.min - p.default) / p.sigma, (p.max - p.default) / p.sigma, loc=p.default, scale=p.sigma))
        return np.array(scaled)

    def unscale(self, scaled):
        """Unscale from the unit hypercube to parameter values. If you want other dists and nested sampling, overwrite this."""
        params = []
        for s, p in zip(scaled, self.get_active_params()):
            if p.prior == "flat":
                params.append(p.min + s * (p.max - p.min))
            else:
                params.append(truncnorm.ppf(s, (p.min - p.default) / p.sigma, (p.max - p.default) / p.sigma, loc=p.default, scale=p.sigma))
        return np.array(params)

    def optimize(self, tol=1.0e-6):
        """Perform local optimisation to try and find the best fit of your model to the dataset loaded in.

        Parameters
        ----------
        tol : float, optional
            Optimisation tolerance

        Returns
        -------
        best_fit_params : dict
            A dictionary mapping parameter names to best fit values
        log_posterior : float
            The value of the best fit log posterior
        """
        self.logger.info("Beginning optimisation!")

        def minimise(scale_params):
            return -self.get_posterior(self.unscale(scale_params))

        bounds = [(0.0, 1.0) for _ in self.get_active_params()]
        res = differential_evolution(minimise, bounds, tol=tol)

        ps = self.unscale(res.x)
        return self.get_param_dict(ps), -res.fun

    def plot_default(self, dataset):
        params = self.get_param_dict(self.get_defaults())
        self.set_data(dataset.get_data())
        self.plot(params)

    @abstractmethod
    def plot(self, params, smooth_params=None, figname=None):
        """Plots the predictions given some input parameter dictionary."""
        pass

    def sanity_check(self, dataset, niter=200, maxiter=10000, figname=None, plot=True):
        import timeit

        print(f"Using dataset {str(dataset)}")
        data = dataset.get_data()
        self.set_data(data)

        p = self.get_defaults()
        p_dict = self.get_param_dict(p)
        posterior = self.get_posterior(p)
        print(self.get_active_params())
        print(f"Posterior {posterior:0.3f} for defaults {dict(p_dict)}")

        assert not np.isnan(posterior), "Posterior should not be nan"

        def timing():
            params = self.get_raw_start()
            posterior = self.get_posterior(params)

        print("Model posterior takes on average, %.2f milliseconds" % (timeit.timeit(timing, number=niter) * 1000 / niter))

        print("Starting model optimisation. This may take some time.")
        p, minv = self.optimize()

        print(f"Model optimisation with value {minv:0.3f} has parameters {dict(p)}")
        if not self.isotropic:
            print(f"\\alpha_{{||}}, \\alpha_{{\\perp}} = ", self.get_alphas(p["alpha"], p["epsilon"]))

        if plot:
            print("Plotting model and data")
            return p, self.plot(p, figname=figname)

        return p


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_power_Beutler2017.py")
    print("bao_power_Ding2018.py")
    print("bao_power_Noda2019.py")
    print("bao_power_Seo2016.py")
    print("bao_correlation_Ross2017.py")
    print("bao_correlation_Ding2018.py")
    print("bao_correlation_Seo2016.py")
