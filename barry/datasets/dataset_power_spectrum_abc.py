import os
import logging
import inspect
import pickle
from abc import ABC

import numpy as np
from scipy.linalg import block_diag

from barry.datasets.dataset import Dataset, MultiDataset
from barry.utils import break2d_into_blocks, break_matrix_and_get_blocks, break_vector_and_get_blocks


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
        fit_poles=(0,),
        data_location=None,
    ):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = (
            os.path.normpath(current_file + f"/../data/{filename}") if data_location is None else data_location + f"/{filename}"
        )
        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.postprocess = postprocess
        if postprocess is not None and not self.isotropic:
            raise NotImplementedError("Postprocessing (i.e., BAOExtractor) not implemented for anisotropic fits")

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)
        name = name or self.data_obj["name"] + " Recon" if recon else self.data_obj["name"] + " Prerecon"
        super().__init__(name, isotropic=isotropic, recon=recon, realisation=realisation)

        self.ndata = self.data_obj["n_data"] if "n_data" in self.data_obj else 1
        self.cosmology = self.data_obj["cosmology"]
        dataname = "post-recon data" if recon else "pre-recon data"
        self.true_data = self.data_obj[dataname] if dataname in self.data_obj else None
        self.mock_data = self.data_obj["post-recon mocks"] if recon else self.data_obj["pre-recon mocks"]
        self.reduce_cov_factor = reduce_cov_factor

        self.poles = [int(c.replace("pk", "")) for c in (self.true_data or self.mock_data)[0].columns if c.startswith("pk")]
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

        if step_size is None:
            self.step_size = list(self.data_obj["winfit"].keys())[0]

        # Rebin the data and mocks as necessary
        rebinned = [self._rebin_data(df) for df in self.true_data] if self.true_data is not None else None
        self.ks = np.split(rebinned[0][0], self.ndata)[0] if rebinned is not None else None
        self.true_data = [x[1] for x in rebinned] if rebinned is not None else None

        rebinned = [self._rebin_data(df) for df in self.mock_data] if self.mock_data is not None else None
        self.ks = np.split(rebinned[0][0], self.ndata)[0] if rebinned is not None else self.ks
        self.mock_data = [x[1] for x in rebinned] if rebinned is not None else None

        self._load_winfit()
        self._load_winpk_file()
        self.m_transform = None
        self.m_w_transform = None
        self.m_w_mask = None
        if not self.isotropic:
            self._load_comp_file()

        self.icov, self.icov_w, self.icov_mw, self.icov_ww, self.icov_mww, self.icov_mwmw = None, None, None, None, None, None
        self.cov, self.cov_fit, self.corr, self.data = None, None, None, None
        self.set_realisation(realisation)
        self.set_cov(fake_diag=fake_diag)

    def set_realisation(self, realisation):
        if realisation is None:
            self.logger.info(f"Loading mock average")
            assert self.mock_data is not None, "Passing in None for the realisations means the mock means, but you have no mocks!"
            self.data = np.array(self.mock_data).mean(axis=0)
            self.realisation = None
        elif str(realisation).lower() == "data":
            assert self.true_data is not None, "Requested data but this dataset doesn't have data set!"
            self.logger.info(f"Loading data")
            self.data = self.true_data[0]
            self.realisation = "data"
        else:
            assert self.mock_data is not None, "You asked for a mock realisation, but this dataset has no mocks!"
            self.logger.info(f"Loading mock {realisation}")
            self.data = self.mock_data[realisation]
            self.realisation = realisation
        return self

    def set_cov(self, fake_diag=False):
        covname = "post-recon cov" if self.recon else "pre-recon cov"
        if covname in self.data_obj:
            if self.data_obj[covname] is not None:
                npoles = len(self.poles)
                nin = len(self.w_mask)
                nout = self.data.shape[0]
                self.cov = np.empty((npoles * nout, npoles * nout))
                w_mask_indices = np.ix_(self.w_mask, self.w_mask)
                for i in range(npoles):
                    iinlow, iinhigh = i * nin, (i + 1) * nin
                    ioutlow, iouthigh = i * nout, (i + 1) * nout
                    for j in range(npoles):
                        jinlow, jinhigh = j * nin, (j + 1) * nin
                        joutlow, jouthigh = j * nout, (j + 1) * nout
                        subset = self.data_obj[covname][iinlow:iinhigh, jinlow:jinhigh][w_mask_indices]
                        self.cov[ioutlow:iouthigh, joutlow:jouthigh] = subset
            else:
                self._compute_cov()
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

        d = np.sqrt(np.diag(self.cov_fit))
        self.corr = self.cov_fit / (d * np.atleast_2d(d).T)
        self.icov = np.linalg.inv(self.cov_fit)
        if self.m_w_transform is not None:
            w_mask_poles = [self.w_mask] * len(self.poles)
            for i, pole in enumerate(self.poles):
                if pole not in self.fit_poles:
                    w_mask_poles[i] = np.zeros(len(self.w_mask), dtype=bool)
            w_mask_poles = np.concatenate(w_mask_poles)
            # self.icov_w = self.w_transform[w_mask_poles, :].T @ self.icov
            # self.icov_mw = self.m_w_transform[w_mask_poles, :].T @ self.icov
            # self.icov_ww = self.icov_w @ self.w_transform[w_mask_poles, :]
            # self.icov_mww = self.icov_mw @ self.w_transform[w_mask_poles, :]
            # self.icov_mwmw = self.icov_mw @ self.m_w_transform[w_mask_poles, :]
            self.m_w_mask = w_mask_poles

    def _compute_cov(self):
        ad = np.array(self.mock_data)
        x0 = np.concatenate([ad[:, :, pole] for pole in self.poles], axis=1)
        self.cov = np.cov(x0.T)
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
        poles = self.poles
        # if self.isotropic:
        #    poles = [p for p in self.poles if p == 0]
        #    assert len(poles) == 1, "Could not find 'pk0' pole"

        k_rebinned, pk0_rebinned, mask = self._agg_data(dataframe, "pk0")
        if self.postprocess is not None:
            pk0_rebinned = self.postprocess(ks=k_rebinned, pk=pk0_rebinned, mask=mask)
        else:
            pk0_rebinned = pk0_rebinned[mask]
        pk_rebinned = np.empty((len(pk0_rebinned), len(poles)))
        pk_rebinned[:, 0] = pk0_rebinned
        for i, pole in enumerate(poles[1:]):
            k_rebinned, pkpole_rebinned, mask = self._agg_data(dataframe, f"pk{pole}")
            pk_rebinned[:, i + 1] = pkpole_rebinned[mask]
        return k_rebinned[mask], pk_rebinned

    def _load_winfit(self):
        self.w_ks_input = self.data_obj["winfit"][self.step_size]["w_ks_input"]
        self.w_transform = self.data_obj["winfit"][self.step_size]["w_transform"]
        self.w_ks_output = self.data_obj["winfit"][self.step_size]["w_ks_output"]
        self.w_mask = np.tile(np.array([np.isclose(x, self.ks).any() for x in self.w_ks_output]), self.ndata)
        self.w_k0_scale = self.data_obj["winfit"][self.step_size]["w_k0_scale"]
        # For isotropic, we'll ignore the contributions from higher order multipoles
        # Not strictly correct, but consistent with most past treatments of the power spectrum.
        if self.isotropic:
            self.w_transform = self.w_transform[: self.ndata * len(self.w_ks_output), : self.ndata * len(self.w_ks_input)]
        self.logger.info(f"Winfit matrix has shape {self.w_transform.shape}")

    def _load_winpk_file(self):
        # data files contain (index, k, pk, nk)
        data = self.data_obj["winpk"]
        if data is None:
            self.w_pk = np.zeros(self.ndata * len(self.w_ks_output))
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
        if self.m_transform is not None:
            self.logger.info(f"Compression matrix has shape {self.m_transform.shape}")
            self.m_w_transform = self.w_transform @ self.m_transform
        else:
            self.m_w_transform = self.w_transform

    def get_data(self):
        d = {
            "ndata": self.ndata,
            "ks_output": self.w_ks_output,
            "ks": self.ks,
            "cov": self.cov,
            "icov": self.icov,
            "icov_m_w": [self.icov_w, self.icov_mw, self.icov_ww, self.icov_mww, self.icov_mwmw],
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
            "poles": np.array(self.poles),
            "fit_poles": self.fit_poles,
            "fit_pole_indices": self.fit_pole_indices,
            "min_k": self.min_k,
            "max_k": self.max_k,
        }

        # Some data has pk0 some has pk0 to pk4
        if self.isotropic:
            d.update({"w_mask": self.w_mask})
            d.update({"m_w_mask": self.w_mask})
            d.update({"pk": self.data[:, 0].reshape(self.ndata, len(self.ks)).flatten()})
        else:
            d.update({"w_mask": np.tile(self.w_mask, len(self.poles))})
            d.update({"m_w_mask": self.m_w_mask})
            d.update({"pk": self.data[:, self.fit_pole_indices].flatten("F")})
        d.update({f"pk{d}": np.split(self.data[:, i], self.ndata) for i, d in enumerate(self.poles)})
        return [d]


if __name__ == "__main__":
    print("Calling a Generic model class as main does not do anything. Try running the Concrete class: ")
    print("dataset_power_spectrum.py")
