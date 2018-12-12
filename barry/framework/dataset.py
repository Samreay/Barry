from abc import ABC
import logging


class Dataset(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger("barry")

    def get_data(self):
        raise NotImplementedError("Please implement get_data")
