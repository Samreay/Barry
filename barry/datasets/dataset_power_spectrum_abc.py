import os
import logging
import inspect
import pickle
from abc import ABC

import numpy as np

from barry.datasets.dataset import Dataset, MultiDataset


class PowerSpectrum(Dataset, ABC):
    def __init__(
        self,
        filename,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=True,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../data/{filename}")
        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.recon = recon
        self.isotropic = isotropic
        self.postprocess = postprocess
        if postprocess is not None and not self.isotropic:
            raise NotImplementedError("Postprocessing (i.e., BAOExtractor) not implemented for anisotropic fits")

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)
        name = name or self.data_obj["name"] + " Recon" if recon else self.data_obj["name"] + " Prerecon"
        super().__init__(name)

        self.cosmology = self.data_obj["cosmology"]
        dataname = "post-recon data" if recon else "pre-recon data"
        self.true_data = self.data_obj[dataname] if dataname in self.data_obj else None
        self.mock_data = self.data_obj["post-recon mocks"] if recon else self.data_obj["pre-recon mocks"]
        self.reduce_cov_factor = reduce_cov_factor
        if self.reduce_cov_factor == -1:
            self.reduce_cov_factor = len(self.mock_data)
            self.logger.info(f"Setting reduce_cov_factor to {self.reduce_cov_factor}")

        # Some data is just a single set of measurements and a covariance matrix so the number of mocks must be specified
        # Otherwise we can work out how many mocks by counting the number of data vectors.
        if num_mocks is None:
            self.num_mocks = len(self.mock_data)
        else:
            self.num_mocks = num_mocks

        if step_size is None:
            self.step_size = list(self.data_obj["winfit"].keys())[0]

        # Rebin the data and mocks as necessary
        rebinned = [self._rebin_data(df) for df in self.true_data] if self.true_data is not None else None
        self.ks = rebinned[0][0] if rebinned is not None else None
        self.true_data = [x[1] for x in rebinned] if rebinned is not None else None

        rebinned = [self._rebin_data(df) for df in self.mock_data] if self.mock_data is not None else None
        self.ks = rebinned[0][0] if rebinned is not None else self.ks
        self.mock_data = [x[1] for x in rebinned] if rebinned is not None else None

        self._load_winfit()
        self._load_winpk_file()
        self.m_transform = None
        self.m_w_transform = None
        if not self.isotropic:
            self._load_comp_file()

        self.cov, self.cov_fit, self.corr, self.icov, self.data = None, None, None, None, None
        self.set_realisation(realisation)
        self.set_cov(fake_diag=fake_diag)

    def set_realisation(self, realisation):
        if realisation is None:
            self.logger.info(f"Loading mock average")
            self.data = np.array(self.mock_data).mean(axis=0)
        elif type(realisation) is int and realisation < 0:
            self.logger.info(f"Loading data")
            self.data = self.true_data[0]
        elif str(realisation).lower() == "data":
            self.logger.info(f"Loading data")
            self.data = self.true_data[0]
        else:
            self.logger.info(f"Loading mock {realisation}")
            self.data = self.mock_data[realisation]

    def set_cov(self, fake_diag=False):
        covname = "post-recon cov" if self.recon else "pre-recon cov"
        if covname in self.data_obj:
            npoles = 1 if self.isotropic else 5
            nin = len(self.w_mask)
            nout = self.data.shape[0]
            self.cov = np.empty((npoles * nout, npoles * nout))
            for i in range(npoles):
                iinlow, iinhigh = i * nin, (i + 1) * nin
                ioutlow, iouthigh = i * nout, (i + 1) * nout
                for j in range(npoles):
                    jinlow, jinhigh = j * nin, (j + 1) * nin
                    joutlow, jouthigh = j * nout, (j + 1) * nout
                    self.cov[ioutlow:iouthigh, joutlow:jouthigh] = self.data_obj[covname][iinlow:iinhigh, jinlow:jinhigh][np.ix_(self.w_mask, self.w_mask)]
        else:
            self._compute_cov()
        if fake_diag:
            self.cov_fit = np.diag(np.diag(self.cov_fit))
        self.cov /= self.reduce_cov_factor
        self.cov_fit /= self.reduce_cov_factor
        v = np.diag(self.cov_fit @ np.linalg.inv(self.cov_fit))
        if not np.all(np.isclose(v, 1)):
            self.logger.error("ERROR, setting an inappropriate covariance matrix that is almost singular!!!!")
            self.logger.error(f"These should all be 1: {v}")
        d = np.sqrt(np.diag(self.cov))
        self.corr = self.cov / (d * np.atleast_2d(d).T)
        self.icov = np.linalg.inv(self.cov_fit)

    def _compute_cov(self):
        ad = np.array(self.mock_data)
        if self.isotropic:
            x0 = ad[:]
            x0fit = x0
        else:
            x0 = np.concatenate([ad[:, :, 0], ad[:, :, 1], ad[:, :, 2], ad[:, :, 3], ad[:, :, 4]], axis=1)
            x0fit = np.concatenate([ad[:, :, 0], ad[:, :, 2], ad[:, :, 4]], axis=1)
        self.cov = np.cov(x0.T)
        self.cov_fit = np.cov(x0fit.T)
        self.logger.info(f"Computed cov {self.cov.shape}")

    def _agg_data(self, dataframe, pole):

        k = dataframe["k"].values
        pk = dataframe[pole].values
        if self.step_size == 1:
            k_rebinned = k
            pk_rebinned = pk
        else:
            add = k.size % self.step_size
            weight = dataframe["nk"].values
            if add:
                to_add = self.step_size - add
                k = np.concatenate((k, [k[-1]] * to_add))
                pk = np.concatenate((pk, [pk[-1]] * to_add))
                weight = np.concatenate((weight, [0] * to_add))
            k = k.reshape((-1, self.step_size))
            pk = pk.reshape((-1, self.step_size))
            weight = weight.reshape((-1, self.step_size))
            # Take the average of every group of step_size rows to rebin
            k_rebinned = np.average(k, axis=1)
            pk_rebinned = np.average(pk, axis=1, weights=weight)

        mask = (k_rebinned >= self.min_k) & (k_rebinned <= self.max_k)
        return k_rebinned, pk_rebinned, mask

    def _rebin_data(self, dataframe):
        poles = ["pk0"] if self.isotropic else ["pk0", "pk1", "pk2", "pk3", "pk4"]
        k_rebinned, pk0_rebinned, mask = self._agg_data(dataframe, "pk0")
        if self.postprocess is not None:
            pk0_rebinned = self.postprocess(ks=k_rebinned, pk=pk0_rebinned, mask=mask)
        else:
            pk0_rebinned = pk0_rebinned[mask]
        if self.isotropic:
            pk_rebinned = pk0_rebinned
        else:
            pk_rebinned = np.empty((len(pk0_rebinned), len(poles)))
            pk_rebinned[:, 0] = pk0_rebinned
            for i, pole in enumerate(poles[1:]):
                k_rebinned, pkpole_rebinned, mask = self._agg_data(dataframe, pole)
                pk_rebinned[:, i + 1] = pkpole_rebinned[mask]
        return k_rebinned[mask], pk_rebinned

    def _load_winfit(self):
        self.w_ks_input = self.data_obj["winfit"][self.step_size]["w_ks_input"]
        self.w_transform = self.data_obj["winfit"][self.step_size]["w_transform"]
        self.w_ks_output = self.data_obj["winfit"][self.step_size]["w_ks_output"]
        self.w_mask = np.array([np.isclose(x, self.ks).any() for x in self.w_ks_output])
        self.w_k0_scale = self.data_obj["winfit"][self.step_size]["w_k0_scale"]
        # For isotropic, we'll ignore the contributions from higher order multipoles
        # Not strictly correct, but consistent with most past treatments of the power spectrum.
        if self.isotropic:
            self.w_transform = self.w_transform[: len(self.w_ks_output), : len(self.w_ks_input)].T
        self.logger.info(f"Winfit matrix has shape {self.w_transform.shape}")

    def _load_winpk_file(self):
        # data files contain (index, k, pk, nk)
        data = self.data_obj["winpk"]
        if data is None:
            self.w_pk = np.zeros(len(self.w_ks_output))
        else:
            if self.step_size == 1:
                self.w_pk = data[:, 2]
            else:
                pk = data[:, 2]
                weight = data[:, 3]

                add = pk.size % self.step_size
                if add:
                    to_add = self.step_size - add
                    pk = np.concatenate((pk, [pk[-1]] * to_add))
                    weight = np.concatenate((weight, [0] * to_add))
                pk = pk.reshape((-1, self.step_size))
                weight = weight.reshape((-1, self.step_size))

                # Take the average of every group of step_size rows to rebin
                self.w_pk = np.average(pk, axis=1, weights=weight)
        self.logger.info(f"Loaded winpk with shape {self.w_pk.shape}")

    def _load_comp_file(self):
        self.m_transform = self.data_obj["m_mat"]
        self.m_w_transform = self.w_transform @ self.m_transform
        self.logger.info(f"Compression matrix has shape {self.m_transform.shape}")

    def get_data(self):
        d = {
            "ks_output": self.w_ks_output,
            "ks": self.ks,
            "cov": self.cov,
            "icov": self.icov,
            "ks_input": self.w_ks_input,
            "w_scale": self.w_k0_scale,
            "w_transform": self.w_transform,
            "w_pk": self.w_pk,
            "corr": self.corr,
            "name": self.name,
            "cosmology": self.cosmology,
            "num_mocks": self.num_mocks,
            "isotropic": self.isotropic,
            "m_transform": self.m_transform,
            "w_m_transform": self.m_w_transform,
        }

        # Some data has pk0 some has pk0 to pk4
        if self.isotropic:
            d.update({"w_mask": self.w_mask})
            d.update({"pk0": self.data})
            d.update({"pk": self.data})
        else:
            d.update({"w_mask": np.tile(self.w_mask, 5)})
            d.update({"pk": np.concatenate([self.data[:, 0], self.data[:, 2], self.data[:, 4]])})
            d.update({"pk0": self.data[:, 0], "pk1": self.data[:, 1], "pk2": self.data[:, 2], "pk3": self.data[:, 3], "pk4": self.data[:, 4]})

        return [d]


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running the Concrete class: ")
    print("dataset_power_spectrum.py")
