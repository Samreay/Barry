import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from numpy.random import uniform
import numpy as np
from scipy.special import loggamma
from scipy.optimize import basinhopping
from enum import Enum, unique
from dataclasses import dataclass


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

    def __init__(self, name, postprocess=None, correction=None):
        """ Create a new model.

        Parameters
        ----------
        name : str
            The name of the model
        postprocess : `Postprocess` class, optional
            The postprocessing class to apply to the model before computing the likelihood. This is not applied automatically.
        correction : `Correction` class, optional
            Correction to apply to the likelihood. Applied automatically. Defaults to `Correction.SELLENTIN`
        """
        self.name = name
        self.logger = logging.getLogger("barry")
        self.data = None
        self.params = []
        self.fix_params = []
        self.param_dict = {}
        self.postprocess = postprocess
        if correction is None:
            correction = Correction.SELLENTIN
        self.correction = correction
        self.correction_data = {}  # Empty dict to store correction specific data for speeding up computation
        assert isinstance(self.correction, Correction), "Correction should be an enum of Correction"
        self.logger.info(f"Created model {name} of {self.__class__.__name__} with correction {correction} and postprocess {str(postprocess)}")

    def get_name(self):
        return self.name

    def set_data(self, data):
        if not isinstance(data, list):
            data = [data]
        self.data = data

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

        if self.correction is Correction.HARTLAP:  # From Hartlap 2007
            chi2 *= (num_mocks - diff.shape - 2) / (num_mocks - 1)

        if self.correction is Correction.SELLENTIN:  # From Sellentin 2016
            key = f"{num_mocks}_{num_params}"
            if key not in self.correction_data:
                self.correction_data[key] = (
                    loggamma(num_mocks / 2).real - (num_params / 2) * np.log(np.pi * (num_mocks - 1)) - loggamma((num_mocks - num_params) * 0.5).real
                )
            c_p = self.correction_data[key]
            log_likelihood = c_p - (num_mocks / 2) * np.log(1 + chi2 / (num_mocks - 1))
            return log_likelihood
        else:
            return -0.5 * chi2

    @abstractmethod
    def get_likelihood(self, params, data):
        raise NotImplementedError("You need to set your likelihood")

    def get_raw_start(self):
        """ Gets a uniformly distributed starting point between parameter min and max constraints """
        start_random = np.array([uniform(x.min, x.max) for x in self.get_active_params()])
        return start_random

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
            minimise, self.scale(start_close), niter_success=3, niter=30, stepsize=0.05, minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 600}}
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
                minimise, self.scale(start), niter_success=10, niter=niter, stepsize=0.05, minimizer_kwargs={"method": m, "options": {"maxiter": maxiter}}
            )
            fs.append(res.fun)
            xs.append(res.x)
        fs = np.array(fs)
        ps = self.unscale(xs[fs.argmin()])
        return self.get_param_dict(ps), fs.min()

    @abstractmethod
    def plot(self, params, smooth_params=None):
        """ Plots the predictions given some input parameter dictionary. """
        pass
