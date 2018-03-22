from abc import ABC
import logging


class Simulation(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def get_data(self):
        raise NotImplementedError("Please implement get_data")
