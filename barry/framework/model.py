import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from numpy.random import uniform
import numpy as np


Param = namedtuple('Param', ['name', 'label', 'min', 'max'])


class Model(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.params = []

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def add_param(self, name, label, min, max):
        self.params.append(Param(name, label, min, max, ))

    def get_labels(self):
        return [x.label for x in self.params]

    def get_extents(self):
        return [(x.min, x.max) for x in self.params]

    def get_prior(self, *params):
        """ The prior, implemented as a flat prior by default"""
        for val, param in zip(params, self.params):
            if val < param.min or val > param.max:
                return -np.inf
        return 0

    @abstractmethod
    def get_likelihood(self, *params):
        raise NotImplementedError("You need to set your likelihood")

    def get_start(self):
        return [uniform(x.min, x.max) for x in self.params]

    def get_posterior(self, *params):
        prior = self.get_prior(*params)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.get_likelihood(*params)
