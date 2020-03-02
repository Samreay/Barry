import os
import pickle
import logging
import inspect
from abc import ABC

import numpy as np

from barry.datasets.dataset import Dataset


class CorrelationFunction(Dataset, ABC):
    def __init__(self, filename, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, num_mocks=None, realisation=None, isotropic=False):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../data/{filename}")
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.recon = recon
        self.isotropic = isotropic

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)
        name = name or self.data_obj["name"] + " Recon" if recon else self.data_obj["name"] + " Prerecon"
        super().__init__(name)

        self.cosmology = self.data_obj["cosmology"]
        self.all_data = self.data_obj["post-recon"] if recon else self.data_obj["pre-recon"]
        self.reduce_cov_factor = reduce_cov_factor
        if self.reduce_cov_factor == -1:
            self.reduce_cov_factor = len(self.all_data)
            self.logger.info(f"Setting reduce_cov_factor to {self.reduce_cov_factor}")

        # Some data is just a single set of measurements and a covariance matrix so the number of mocks must be specified
        # Otherwise we can work out how many mocks by counting the number of data vectors.
        if num_mocks is None:
            self.num_mocks = len(self.all_data)
        else:
            self.num_mocks = num_mocks

        self.cov, self.icov, self.data, self.mask = None, None, None, None
        self.set_realisation(realisation)
        self.set_cov()

    def set_realisation(self, realisation):
        if realisation is None:
            self.logger.info(f"Loading data average")
            self.data = np.array(self.all_data).mean(axis=0)
        else:
            self.logger.info(f"Loading realisation {realisation}")
            self.data = self.all_data[realisation]
        self.mask = (self.data[:, 0] >= self.min_dist) & (self.data[:, 0] <= self.max_dist)
        self.data = self.data[self.mask, :]

    def set_cov(self):
        covname = "post-recon cov" if self.recon else "pre-recon cov"
        if covname in self.data_obj:
            npoles = 1 if self.isotropic else 2
            nin = len(self.mask)
            nout = self.data.shape[0]
            self.cov = np.empty((npoles * nout, npoles * nout))
            for i in range(npoles):
                iinlow, iinhigh = i * nin, (i + 1) * nin
                ioutlow, iouthigh = i * nout, (i + 1) * nout
                for j in range(npoles):
                    jinlow, jinhigh = j * nin, (j + 1) * nin
                    joutlow, jouthigh = j * nout, (j + 1) * nout
                    self.cov[ioutlow:iouthigh, joutlow:jouthigh] = self.data_obj[covname][iinlow:iinhigh, jinlow:jinhigh][np.ix_(self.mask, self.mask)]
        else:
            self._compute_cov()
        self.cov /= self.reduce_cov_factor
        self.icov = np.linalg.inv(self.cov)

    def _compute_cov(self):
        ad = np.array(self.all_data)
        if self.isotropic:
            x0 = ad[:, self.mask, 1]
        else:
            x0 = np.concatenate([ad[:, self.mask, 1], ad[:, self.mask, 2]], axis=1)
        self.cov = np.cov(x0.T)
        self.logger.info(f"Computed cov {self.cov.shape}")

    def get_data(self):
        d = {
            "dist": self.data[:, 0],
            "cov": self.cov,
            "icov": self.icov,
            "name": self.name,
            "cosmology": self.cosmology,
            "num_mocks": self.num_mocks,
            "isotropic": self.isotropic,
        }

        # Some data has xi0 some has xi0+xi2
        d.update({"xi0": self.data[:, 1]})
        if self.isotropic:
            d.update({"xi": self.data[:, 1]})
        else:
            d.update({"xi": np.concatenate([self.data[:, 1], self.data[:, 2]])})
            d.update({"xi2": self.data[:, 2]})

        return [d]


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running the Concrete class: ")
    print("dataset_correlation_function.py")
