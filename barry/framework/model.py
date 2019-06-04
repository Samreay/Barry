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
        for pname, val in params:
            if val < self.param_dict[pname].min or val > self.param_dict[pname].max:
                return -np.inf
        return 0

    @abstractmethod
    def get_likelihood(self, params):
        raise NotImplementedError("You need to set your likelihood")

    def get_start(self):
        return [uniform(x.min, x.max) for x in self.get_active_params()]

    def get_posterior(self, params):
        ps = OrderedDict([(p.name, v) for p, v in zip(self.get_active_params(), params)])
        ps.update({(p.name, p.default) for p in self.get_inactive_params()})
        prior = self.get_prior(ps)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.get_likelihood(ps)
