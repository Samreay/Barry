import logging
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
from numpy.random import uniform
import numpy as np


Param = namedtuple('Param', ['name', 'label', 'min', 'max', 'default'])


class Model(ABC):
    def __init__(self, name, postprocess=None):
        self.name = name
        self.logger = logging.getLogger("barry")
        self.data = None
        self.params = []
        self.fix_params = []
        self.param_dict = {}
        self.postprocess = postprocess

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def add_param(self, name, label, min, max, default):
        p = Param(name, label, min, max, default)
        self.params.append(p)
        self.param_dict[name] = p

    def set_fix_params(self, params):
        if params is None:
            params = []
        self.fix_params = params

    def get_param(self, dic, name):
        return dic.get(name, self.get_default(name))

    def get_active_params(self):
        return [p for p in self.params if p.name not in self.fix_params]

    def get_inactive_params(self):
        return [p for p in self.params if p.name in self.fix_params]

    def get_default(self, name):
        return self.param_dict[name].default

    def get_labels(self):
        return [x.label for x in self.get_active_params()]

    def get_extents(self):
        return [(x.min, x.max) for x in self.get_active_params()]

    def get_prior(self, params):
        """ The prior, implemented as a flat prior by default"""
        for pname, val in params.items():
            if val < self.param_dict[pname].min or val > self.param_dict[pname].max:
                return -np.inf
        return 0

    @abstractmethod
    def get_likelihood(self, params):
        raise NotImplementedError("You need to set your likelihood")

    def get_start(self):
        return [uniform(x.min, x.max) for x in self.get_active_params()]

    def get_start_scaled(self):
        return self.scale(self.get_start())

    def get_param_dict(self, params):
        ps = OrderedDict([(p.name, v) for p, v in zip(self.get_active_params(), params)])
        ps.update({(p.name, p.default) for p in self.get_inactive_params()})
        return ps

    def get_posterior_scaled(self, scaled):
        return self.get_posterior(self.unscale(scaled))

    def get_posterior(self, params):
        ps = self.get_param_dict(params)
        prior = self.get_prior(ps)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.get_likelihood(ps)

    def scale(self, params):
        scaled = np.array([(s - p.min) / (p.max - p.min) for s, p in zip(params, self.get_active_params())])
        return scaled

    def unscale(self, scaled):
        params = [p.min + s * (p.max - p.min) for s, p in zip(scaled, self.get_active_params())]
        return params

    def optimize(self, niter=10, close_default=5):
        from scipy.optimize import minimize

        def minimise(scale_params):
            return -self.get_posterior(self.unscale(scale_params))

        fs = []
        xs = []
        methods = ['Nelder-Mead']
        for i in range(niter):
            for m in methods:
                start = np.array(self.get_start())
                if close_default:
                    start = [(s + p.default * close_default) / (1 + close_default) for s, p in zip(start, self.get_active_params())]
                bounds = [(0, 1) for p in self.get_active_params()]
                res = minimize(minimise, self.scale(start), method=m, bounds=bounds, options={"maxiter": 1000})
                fs.append(res.fun)
                xs.append(res.x)
        fs = np.array(fs)
        ps = self.unscale(xs[fs.argmin()])
        return self.get_param_dict(ps), fs.min()

    @abstractmethod
    def plot(self, *params):
        pass
