from abc import ABC
import logging


class Data(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)