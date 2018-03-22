import logging
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def get_name(self):
        return self.name

    @abstractmethod
    def get_prior(self, data, params):
        raise NotImplementedError("You need to set your prior")

    @abstractmethod
    def get_likelihood(self, data, params):
        raise NotImplementedError("You need to set your likelihood")
