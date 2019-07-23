import os
import logging
import inspect
import pickle
import numpy as np
from barry.framework.dataset import Dataset


class MockPowerSpectrum(Dataset):
    def __init__(self, filename, realisation=None, min_k=0.02, max_k=0.30, step_size=None, recon=True,
                 reduce_cov_factor=1, postprocess=None, apply_hartlap_correction=False,
                 fake_diag=False):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../../data/{filename}")

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)

        super().__init__(self.data_obj["name"])

        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.recon = recon
        self.realisation = realisation
        self.postprocess = postprocess
        self.reduce_cov_factor = reduce_cov_factor

        self.cosmology = self.data_obj["cosmology"]
        self.all_data = self.data_obj["post-recon"] if recon else self.data_obj["pre-recon"]

        if step_size is None:
            self.step_size = self.data_obj["winfit"].keys()[0]

        self.rebinned = [self._rebin_data(df) for df in self.all_data]
        self.ks = self.rebinned[0][0]
        self.pks_all = [x[1] for x in self.rebinned]

        if self.realisation is None:
            self.logger.info(f"Loading data average")
            self.data = self._get_data_avg()
        else:
            self.logger.info(f"Loading realisation {realisation}")
            self.data = self.pks_all[realisation]

        self._load_winfit()
        self._load_winpk_file()
        self.logger.debug(f"Computing cov")
        self.set_cov(self._compute_cov(), apply_correction=apply_hartlap_correction, fake_diag=fake_diag)

    def set_realisation(self, index):
        self.data = self.pks_all[index]

    def set_cov(self, cov, apply_correction=False, fake_diag=False):
        self.logger.info(f"Computed cov {cov.shape}")
        if self.reduce_cov_factor != 1:
            self.logger.info(f"Reducing covariance by factor of {self.reduce_cov_factor}")
            cov /= self.reduce_cov_factor
        if fake_diag:
            cov = np.diag(np.diag(cov))
        self.cov = cov

        v = np.diag(cov @ np.linalg.inv(cov))
        if not np.all(np.isclose(v, 1)):
            self.logger.error("ERROR, setting an inappropriate covariance matrix that is almost singular!!!!")
            self.logger.error(f"These should all be 1: {v}")
        d = np.sqrt(np.diag(self.cov))
        self.corr = self.cov / (d * np.atleast_2d(d).T)
        if apply_correction:
            self.correction_factor = (len(self.all_data) - self.corr.shape[0] - 2) / (len(self.all_data) - 1)
            self.logger.info(f"icov correction factor is {self.correction_factor:0.5f}, from Hartlap 2007")
        else:
            self.correction_factor = 1
        self.icov = np.linalg.inv(self.cov) * self.correction_factor

    def _compute_cov(self):
        pks = np.array(self.pks_all)
        cov = np.cov(pks.T)
        return cov

    def _get_data_avg(self):
        return np.array(self.pks_all).mean(axis=0)

    def _agg_data(self, dataframe):

        k = dataframe["k"].values
        pk = dataframe["pk"].values
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
        k_rebinned, pk_rebinned, mask = self._agg_data(dataframe)
        if self.postprocess is not None:
            pk_rebinned = self.postprocess(ks=k_rebinned, pk=pk_rebinned, mask=mask)
        else:
            pk_rebinned = pk_rebinned[mask]
        return k_rebinned[mask], pk_rebinned

    def _load_winfit(self):
        self.w_ks_input = self.data_obj["winfit"][self.step_size]["w_ks_input"]
        self.w_k0_scale = self.data_obj["winfit"][self.step_size]["w_k0_scale"]
        self.w_transform = self.data_obj["winfit"][self.step_size]["w_transform"]
        self.w_ks_output = self.data_obj["winfit"][self.step_size]["w_ks_output"]
        self.w_mask = np.array([np.isclose(x, self.ks).any() for x in self.w_ks_output])
        self.logger.info(f"Winfit matrix has shape {self.w_transform.shape}")

    def _load_winpk_file(self):
        # data files contain (index, k, pk, nk)
        data = self.data_obj["winpk"]
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

    def get_data(self):
        return {
            "ks_output": self.w_ks_output,
            "ks": self.ks,
            "pk": self.data,
            "cov": self.cov,
            "icov": self.icov,
            "ks_input": self.w_ks_input,
            "w_scale": self.w_k0_scale,
            "w_transform": self.w_transform,
            "w_pk": self.w_pk,
            "w_mask": self.w_mask,
            "corr": self.corr,
            "name": self.name
        }


class MockSDSSPowerSpectrum(MockPowerSpectrum):
    def __init__(self, realisation=None, apply_hartlap_correction=False, fake_diag=False, recon=False, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=5, postprocess=None):
        super().__init__("sdss_dr7_pk.pkl", min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, postprocess=postprocess, realisation=realisation, apply_hartlap_correction=apply_hartlap_correction, fake_diag=fake_diag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    # Some basic checks for data we expect to be there
    dataset = MockSDSSPowerSpectrum(recon=False)
    # print(dataset.all_data)
    data = dataset.get_data()
    # print(data["ks"])
    #
    import matplotlib.pyplot as plt
    import numpy as np
    plt.errorbar(data["ks"], data["ks"]*data["pk"], yerr=data["ks"]*np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=1)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=2, recon=False)
    # MockAveragePowerSpectrum(step_size=5, recon=False)
