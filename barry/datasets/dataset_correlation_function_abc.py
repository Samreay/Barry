import os
import pickle
import logging
import inspect
from abc import ABC

import numpy as np

from barry.datasets.dataset import Dataset
from barry.utils import break_matrix_and_get_blocks


class CorrelationFunction(Dataset, ABC):
    def __init__(
        self,
        filename,
        name=None,
        min_dist=30,
        max_dist=200,
        recon=True,
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=(0,),
    ):

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
        dataname = "post-recon data" if recon else "pre-recon data"
        self.true_data = self.data_obj[dataname] if dataname in self.data_obj else None
        self.mock_data = self.data_obj["post-recon mocks"] if recon else self.data_obj["pre-recon mocks"]
        self.reduce_cov_factor = reduce_cov_factor

        self.poles = [int(c.replace("xi", "")) for c in (self.true_data or self.mock_data)[0].columns if c.startswith("xi")]
        if fit_poles is None:
            if isotropic:
                self.fit_poles = [0]
            else:
                self.fit_poles = [x for x in self.poles if x % 2 == 0]
        else:
            self.fit_poles = fit_poles
        self.fit_pole_indices = np.where([i in self.fit_poles for i in self.poles])[0]

        if self.reduce_cov_factor == -1:
            self.reduce_cov_factor = len(self.mock_data)
            self.logger.info(f"Setting reduce_cov_factor to {self.reduce_cov_factor}")

        # Some data is just a single set of measurements and a covariance matrix so the number of mocks must be specified
        # Otherwise we can work out how many mocks by counting the number of data vectors.
        if num_mocks is None:
            self.num_mocks = 0 if self.mock_data is None else len(self.mock_data)
        else:
            self.num_mocks = num_mocks

        # Reformat the data and mocks as necessary
        rebinned = [self._agg_data(df) for df in self.true_data] if self.true_data is not None else None
        self.ss = rebinned[0][0] if rebinned is not None else None
        self.true_data = [x[1] for x in rebinned] if rebinned is not None else None

        rebinned = [self._agg_data(df) for df in self.mock_data] if self.mock_data is not None else None
        self.ss = rebinned[0][0] if rebinned is not None else self.ss
        self.mock_data = [x[1] for x in rebinned] if rebinned is not None else None

        self.cov, self.icov, self.data = None, None, None
        self.set_realisation(realisation)
        self.set_cov(fake_diag=fake_diag)

    def set_realisation(self, realisation):
        if realisation is None:
            self.logger.info(f"Loading mock average")
            assert self.mock_data is not None, "Passing in None for the realisations means the mock means, but you have no mocks!"
            self.data = np.array(self.mock_data).mean(axis=0)
        elif str(realisation).lower() == "data":
            assert self.true_data is not None, "Requested data but this dataset doesn't have data set!"
            self.logger.info(f"Loading data")
            self.data = self.true_data[0]
        else:
            assert self.mock_data is not None, "You asked for a mock realisation, but this dataset has no mocks!"
            self.logger.info(f"Loading mock {realisation}")
            self.data = self.mock_data[realisation]
        # self.mask = (self.ss >= self.min_dist) & (self.ss <= self.max_dist)
        # print(np.shape(self.data), np.shape(self.mask))
        # self.ss = self.ss[self.mask]
        # self.data = self.data[self.mask, :]

    def _agg_data(self, dataframe):

        poles = self.poles
        ss = dataframe["s"].values
        self.mask = (ss >= self.min_dist) & (ss <= self.max_dist)
        xi0_rebinned = dataframe["xi0"].values[self.mask]
        xi_rebinned = np.empty((len(xi0_rebinned), len(poles)))
        xi_rebinned[:, 0] = xi0_rebinned
        for i, pole in enumerate(poles[1:]):
            xi_rebinned[:, i + 1] = dataframe[f"xi{pole}"].values[self.mask]

        return ss[self.mask], xi_rebinned, self.mask

    def set_cov(self, fake_diag=False):
        covname = "post-recon cov" if self.recon else "pre-recon cov"
        if covname in self.data_obj:
            npoles = len(self.poles)
            nin = len(self.mask)
            nout = self.data.shape[0]
            self.cov = np.empty((npoles * nout, npoles * nout))
            for i in range(npoles):
                iinlow, iinhigh = i * nin, (i + 1) * nin
                ioutlow, iouthigh = i * nout, (i + 1) * nout
                for j in range(npoles):
                    jinlow, jinhigh = j * nin, (j + 1) * nin
                    joutlow, jouthigh = j * nout, (j + 1) * nout
                    self.cov[ioutlow:iouthigh, joutlow:jouthigh] = self.data_obj[covname][iinlow:iinhigh, jinlow:jinhigh][
                        np.ix_(self.mask, self.mask)
                    ]
        else:
            self._compute_cov()

        self.cov_fit = break_matrix_and_get_blocks(self.cov, len(self.poles), self.fit_pole_indices)

        if fake_diag:
            self.cov_fit = np.diag(np.diag(self.cov_fit))
        self.cov /= self.reduce_cov_factor
        self.cov_fit /= self.reduce_cov_factor

        # Run some checks
        v = np.diag(self.cov_fit @ np.linalg.inv(self.cov_fit))
        if not np.all(np.isclose(v, 1)):
            self.logger.error("ERROR, setting an inappropriate covariance matrix that is almost singular!!!!")
            self.logger.error(f"These should all be 1: {v}")

        self.icov = np.linalg.inv(self.cov_fit)

    def _compute_cov(self):
        ad = np.array(self.mock_data)
        if self.isotropic:
            x0 = ad[:, self.mask, 1]
        else:
            x0 = np.concatenate([ad[:, self.mask, 1], ad[:, self.mask, 2]], axis=1)
        self.cov = np.cov(x0.T)
        self.logger.info(f"Computed cov {self.cov.shape}")

    def get_data(self):
        d = {
            "dist": self.ss,
            "cov": self.cov,
            "icov": self.icov,
            "name": self.name,
            "cosmology": self.cosmology,
            "num_mocks": self.num_mocks,
            "isotropic": self.isotropic,
            "poles": self.poles,
            "fit_poles": self.fit_poles,
            "fit_pole_indices": self.fit_pole_indices,
            "min_dist": self.min_dist,
            "max_dist": self.max_dist,
        }

        # Some data has xi0 some has xi0+xi2
        if self.isotropic:
            d.update({"xi": self.data[:, 0]})
        else:
            d.update({"xi": self.data[:, self.fit_pole_indices].T.flatten()})
        d.update({f"xi{d}": self.data[:, i] for i, d in enumerate(self.poles)})

        return [d]


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running the Concrete class: ")
    print("dataset_correlation_function.py")
