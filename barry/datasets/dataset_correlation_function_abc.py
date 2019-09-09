import os
import pickle
import logging
import inspect
from abc import ABC

import numpy as np

from barry.datasets.dataset import Dataset


class CorrelationFunction(Dataset, ABC):
    def __init__(self, filename, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../data/{filename}")
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.recon = recon

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)
        super().__init__(self.data_obj["name"])
        self.cosmology = self.data_obj["cosmology"]

        self.all_data = self.data_obj["post-recon"] if recon else self.data_obj["pre-recon"]
        self.reduce_cov_factor = reduce_cov_factor
        if self.reduce_cov_factor == -1:
            self.reduce_cov_factor = len(self.all_data)
            self.logger.info(f"Setting reduce_cov_factor to {self.reduce_cov_factor}")

        self.cov, self.icov, self.data, self.mask = None, None, None, None
        self.set_realisation(realisation)
        self._compute_cov()

    def set_realisation(self, realisation):
        if realisation is None:
            self.data = np.array(self.all_data).mean(axis=0)
        else:
            self.data = self.all_data[realisation]
        self.mask = (self.data[:, 0] >= self.min_dist) & (self.data[:, 0] <= self.max_dist)
        self.data = self.data[self.mask, :]

    def set_cov(self, cov):
        self.cov = cov / self.reduce_cov_factor
        self.icov = np.linalg.inv(self.cov)

    def _compute_cov(self):
        # TODO: Generalise for other multipoles poles
        ad = np.array(self.all_data)
        if ad.shape[2] > 2:
            x0 = ad[:, self.mask, 2]
        else:
            x0 = ad[:, self.mask, 1]
        cov = np.cov(x0.T)
        self.set_cov(cov)

    def get_data(self):
        d = {"dist": self.data[:, 0], "cov": self.cov, "icov": self.icov, "name": self.name, "cosmology": self.cosmology, "num_mocks": len(self.all_data)}
        if self.data.shape[1] > 2:  # Some data has xi, xi0, xi2, xi4, some only has xi0
            d.update({"xi": self.data[:, 1], "xi0": self.data[:, 2], "xi2": self.data[:, 3], "xi4": self.data[:, 4]})
        else:
            d.update({"xi0": self.data[:, 1]})
        return [d]
