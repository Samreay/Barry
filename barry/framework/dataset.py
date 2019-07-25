from abc import ABC
import logging


class Dataset(ABC):
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger("barry")

    def get_name(self):
        return self.name

    def get_data(self):
        raise NotImplementedError("Please implement get_data")


class MultiDataset(Dataset):
    def __init__(self, name, datasets):
        super().__init__(name)
        self.datasets = datasets

    def get_data(self):
        return [i for d in self.datasets for i in d.get_data()]