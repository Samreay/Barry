import os
import inspect
import logging
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from numpy.random import uniform
import numpy as np
from scipy import integrate
from scipy.integrate import simps
from scipy.special import loggamma, erfc
from scipy.optimize import basinhopping
from enum import Enum, unique
from dataclasses import dataclass


from barry.cosmology.camb_generator import Omega_m_z, getCambGenerator


@dataclass
class Param:
    name: str
    label: str
    min: float
    max: float
    default: float
    active: bool


@unique
class Correction(Enum):
    """ Various corrections that we should apply when computing our likelihood.

    NONE gives no correction.
    HARTLAP implements the chi2 correction given in Hartlap 2007
    SELLENTIN implements the correction from Sellentin 2016

    """

    NONE = 0
    HARTLAP = 1
    SELLENTIN = 2


class Model(ABC):
    """ Abstract model class.

    Implement a version of this that overwrites the `plot` and get_likelihood` methods.

    """

    def __init__(self, name, postprocess=None, correction=None, isotropic=True, marg=False):
        """ Create a new model.

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
            correction = Correction.SELLENTIN
        self.correction = correction
        self.correction_data = {}  # Empty dict to store correction specific data for speeding up computation
        assert isinstance(self.correction, Correction), "Correction should be an enum of Correction"
        self.logger.info(
            f"Created model {name} of {self.__class__.__name__} with correction {correction} and postprocess {str(postprocess)}"
        )

        self.marg = marg
        """if self.marg:
            assert (
                self.correction != Correction.SELLENTIN
            ), "ERROR: SELLENTIN covariance matrix correction not compatible with analytic marginalisation. Switch to Correction.HARTLAP or use marg=false"
            assert self.isotropic == False, "ERROR: analytic marginalisation only supported for anisotropic fits currently"""

    def get_name(self):
        return self.name

    def get_unique_cosmo_name(self):
        """ Unique name used to save out any pregenerated data. """
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
            self.camb = getCambGenerator(
                h0=c["h0"], ob=c["ob"], redshift=c["z"], ns=c["ns"], mnu=c["mnu"], recon_smoothing_scale=c["reconsmoothscale"]
            )
            self.set_default("om", c["om"])
            self.pregen_path = os.path.abspath(os.path.join(self.data_location, self.get_unique_cosmo_name()))
            self.cosmology = c
            if load_pregen:
                self._load_precomputed_data()

    def set_data(self, data):
        """ Sets the models data, including fetching the right cosmology and PT generator.

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

    def add_param(self, name, label, min, max, default):
        p = Param(name, label, min, max, default, name not in self.fix_params)
        self.params.append(p)
        self.param_dict[name] = p

    def set_fix_params(self, params):
        if params is None:
            params = []
        self.fix_params = params
        for p in self.params:
            p.active = p.name not in params

    def get_param(self, dic, name):
        return dic.get(name, self.get_default(name))

    def get_active_params(self):
        """ Returns a list of the active (non-fixed) parameters """
        return [p for p in self.params if p.active]

    def get_inactive_params(self):
        """ Returns a list of the inactive (fixed) parameters"""
        return [p for p in self.params if not p.active]

    def get_default(self, name):
        """ Returns the default value of a given parameter name """
        return self.param_dict[name].default

    def set_default(self, name, default):
        """ Sets the default value for a parameter """
        self.param_dict[name].default = default

    def get_defaults(self):
        """ Returns a list of default values for all active parameters """
        return [x.default for x in self.get_active_params()]

    def get_defaults_dict(self):
        """ Returns a list of default values for all active parameters """
        return {x.name: x.default for x in self.get_active_params()}

    def get_labels(self):
        """ Gets a list of the label for all active parameters """
        return [x.label for x in self.get_active_params()]

    def get_names(self):
        """ Get a list of the names for all active parameters """
        return [x.name for x in self.get_active_params()]

    def get_extents(self):
        """ Gets a list of (min, max) extents for all active parameters """
        return [(x.min, x.max) for x in self.get_active_params()]

    def get_prior(self, params):
        """ The prior, implemented as a flat prior by default.

        Used by the Ensemble and MH samplers, but not by nested sampling methods.

        """
        for pname, val in params.items():
            if val < self.param_dict[pname].min or val > self.param_dict[pname].max:
                return -np.inf
        return 0

    def get_chi2_likelihood(self, diff, icov, num_mocks=None, num_params=None):
        """ Computes the chi2 corrected likelihood.

        Parameters
        ----------
        diff : np.ndarray
            The difference between the model predictions and data observations
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
        chi2 = diff.T @ icov @ diff

        if self.correction in [Correction.HARTLAP, Correction.SELLENTIN]:
            assert (
                num_mocks > 0
            ), "Cannot use HARTLAP  or SELLENTIN correction with covariance not determined from mocks. Set correction to Correction.NONE"
        if self.correction is Correction.HARTLAP:  # From Hartlap 2007
            chi2 *= (num_mocks - len(diff) - 2) / (num_mocks - 1)

        if self.correction is Correction.SELLENTIN:  # From Sellentin 2016
            key = f"{num_mocks}_{num_params}"
            if key not in self.correction_data:
                self.correction_data[key] = (
                    loggamma(num_mocks / 2).real
                    - (num_params / 2) * np.log(np.pi * (num_mocks - 1))
                    - loggamma((num_mocks - num_params) * 0.5).real
                )
            c_p = self.correction_data[key]
            log_likelihood = c_p - (num_mocks / 2) * np.log(1 + chi2 / (num_mocks - 1))
            return log_likelihood
        else:
            if np.random.rand() < 0.01:
                print(chi2)
            return -0.5 * chi2

    def get_chi2_marg_likelihood(self, marg_model, data, icov, num_mocks=None):
        """ Computes the chi2 corrected likelihood.

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
        icov_corr = icov
        if self.correction in [Correction.HARTLAP]:
            assert (
                num_mocks > 0
            ), "Cannot use HARTLAP  or SELLENTIN correction with covariance not determined from mocks. Set correction to Correction.NONE"
        if self.correction is Correction.HARTLAP:  # From Hartlap 2007
            icov_corr = icov * (num_mocks - len(data) - 2) / (num_mocks - 1)

        """F00 = model @ icov_corr @ model
        F01 = model @ icov_corr @ data
        F02 = data @ icov_corr @ data
        F11 = marg_model @ icov_corr @ data
        F12 = marg_model @ icov_corr @ model
        F2 = marg_model @ icov_corr @ marg_model.T
        F2inv = np.linalg.inv(F2)

        A = F00 - F12 @ F2inv @ F12
        B = F12 @ F2inv @ F11 - F01

        chi2 = F11 @ F2inv @ F11 - F02 - np.log(np.linalg.det(F2))
        marg_corr = 0.5 * (B ** 2 / A - np.log(A)) + np.log(erfc(B / np.sqrt(2.0 * A)))

        return 0.5 * chi2 + marg_corr"""

        F02 = data @ icov_corr @ data
        F11 = marg_model @ icov_corr @ data
        F2 = marg_model @ icov_corr @ marg_model.T
        F2inv = np.linalg.inv(F2)
        chi2 = F02 - F11 @ F2inv @ F11
        return -0.5 * (chi2 + np.log(np.linalg.det(F2)))

    def get_raw_start(self):
        """ Gets a uniformly distributed starting point between parameter min and max constraints """
        start_random = np.array([uniform(x.min, x.max) for x in self.get_active_params()])
        return start_random

    def _load_precomputed_data(self):
        if self._needs_precompute():
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

            values = self.precompute(self.camb, om, h0)
            data.append([i, j, values])
        return data

    def get_pregen(self, key, om, h0=None):
        if h0 is None:
            h0 = self.camb.h0
        data = self.pregen[key]
        return self.camb.interpolate(om, h0, data)

    def _needs_precompute(self):
        func2 = getattr(super(type(self), self), self.precompute.__name__)
        return self.precompute.__func__ != func2.__func__

    def precompute(self, camb, om, h0):
        """ A function available for overriding that precomputes values that depend only on the outputs of CAMB.

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
        """ Gets an optimised `n` starting points by calculating a best fit starting point using basinhopping """
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
        """ Gets the number of dimensions (active, free parameters) in the model """
        return len(self.get_active_params())

    def get_start_scaled(self):
        """ Gets a scaled (unit hypercube) optimised starting position."""
        return self.scale(self.get_start())

    def get_param_dict(self, params):
        """ Converts a list of parameter values into a dictionary of parameter values """
        ps = OrderedDict([(p.name, v) for p, v in zip(self.get_active_params(), params)])
        ps.update({(p.name, p.default) for p in self.get_inactive_params()})
        return ps

    def get_posterior_scaled(self, scaled):
        """ Gets the posterior using an input scaled (unit hypercube) location in parameter space"""
        return self.get_posterior(self.unscale(scaled))

    def get_posterior(self, params):
        """ Returns the posterior given a list of param values."""
        ps = self.get_param_dict(params)
        prior = self.get_prior(ps)
        if not np.isfinite(prior):
            return -np.inf
        posterior = prior
        for d in self.data:
            posterior += self.get_likelihood(ps, d)
        return posterior

    def scale(self, params):
        """ Scale parameter values to the unit hypercube. Assumes uniform priors. If you want other dists and nested sampling, overwrite this """
        scaled = np.array([(s - p.min) / (p.max - p.min) for s, p in zip(params, self.get_active_params())])
        return scaled

    def unscale(self, scaled):
        """ Unscale from the unit hypercube to parameter values. Assumes uniform. if you want other dists and nested sampling, overwrite this."""
        params = [p.min + s * (p.max - p.min) for s, p in zip(scaled, self.get_active_params())]
        return params

    def optimize(self, close_default=3, niter=100, maxiter=1000):
        """ Perform local optimiation to try and find the best fit of your model to the dataset loaded in.

        Parameters
        ----------
        close_default : int, optional
            How close to the default values we should start our walk. Higher numbers mean closer to the default.
            Used to compute a weighted avg between the default and a uniformly selected starting point.
        niter : int, optional
            How many iterations to run the `basinhopping` algorithm for.
        maxiter : int, optional
            How many steps each iteration can take in the `basinhopping` algorithm.

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

        fs = []
        xs = []
        methods = ["Nelder-Mead"]
        for m in methods:
            start = np.array(self.get_raw_start())
            if close_default:
                start = [(s + p.default * close_default) / (1 + close_default) for s, p in zip(start, self.get_active_params())]
            res = basinhopping(
                minimise,
                self.scale(start),
                niter_success=10,
                niter=niter,
                stepsize=0.05,
                minimizer_kwargs={"method": m, "options": {"maxiter": maxiter, "fatol": 1.0e-5, "xatol": 1.0e-5}},
            )
            fs.append(res.fun)
            xs.append(res.x)
        fs = np.array(fs)
        ps = self.unscale(xs[fs.argmin()])
        return self.get_param_dict(ps), -fs.min()

    def plot_default(self, dataset):
        params = self.get_param_dict(self.get_defaults())
        self.set_data(dataset.get_data())
        self.plot(params)

    @abstractmethod
    def plot(self, params, smooth_params=None, figname=None):
        """ Plots the predictions given some input parameter dictionary. """
        pass

    def sanity_check(self, dataset, niter=200, maxiter=10000, figname=None):
        import timeit

        print(f"Using dataset {str(dataset)}")
        data = dataset.get_data()
        self.set_data(data)

        p = self.get_defaults()
        p_dict = self.get_param_dict(p)
        posterior = self.get_posterior(p)
        print(f"Posterior {posterior:0.3f} for defaults {dict(p_dict)}")

        assert not np.isnan(posterior), "Posterior should not be nan"

        def timing():
            params = self.get_raw_start()
            posterior = self.get_posterior(params)

        print("Model posterior takes on average, %.2f milliseconds" % (timeit.timeit(timing, number=niter) * 1000 / niter))

        # print("Starting model optimisation. This may take some time.")
        # p, minv = self.optimize(niter=niter, maxiter=maxiter)
        # print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")

        # print("Plotting model and data")
        # self.plot(p, figname=figname)

    def integrate_mu(self, pk2d, mu, isotropic=False):
        pk0 = simps(pk2d, mu, axis=1)
        if isotropic:
            pk2 = None
            pk4 = None
        else:
            pk2 = 3.0 * simps(pk2d * mu ** 2, mu)
            pk4 = 1.125 * (35.0 * simps(pk2d * mu ** 4, mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0)
        return pk0, pk2, pk4


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running one of the Concrete classes: ")
    print("bao_power_Beutler2017.py")
    print("bao_power_Ding2018.py")
    print("bao_power_Noda2019.py")
    print("bao_power_Seo2016.py")
    print("bao_correlation_Beutler2017.py")
    print("bao_correlation_Ding2018.py")
    print("bao_correlation_Seo2016.py")
