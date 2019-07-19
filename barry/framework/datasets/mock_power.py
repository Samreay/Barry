import os
import logging
import inspect
import pickle
import numpy as np
from barry.framework.dataset import Dataset


class MockPowerSpectrum(Dataset):
    def __init__(self, average=True, realisation=0, min_k=0.02, max_k=0.30, step_size=2, recon=True,
                 reduce_cov_factor=1, name="MockPowerSpectrum", postprocess=None, apply_hartlap_correction=False,
                 fake_diag=False, data_dir="taipan_mocks"):
        super().__init__(name)
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../../data/{data_dir}/")
        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.recon = recon
        self.realisation = realisation
        self.average = average
        self.postprocess = postprocess
        self.reduce_cov_factor = reduce_cov_factor

        self.data_filename = os.path.abspath(self.data_location + "/mock_lpow.pkl")

        assert os.path.exists(self.data_filename), f"Cannot find {self.data_filename}"

        self.logger.debug(f"Loading data from {self.data_filename}")
        with open(self.data_filename, "rb") as f:
            self.all_data = pickle.load(f)["post-recon" if recon else "pre-recon"]

        self.rebinned = [self._rebin_data(df) for df in self.all_data]
        self.ks = self.rebinned[0][0]
        self.pks_all = [x[1] for x in self.rebinned]

        if self.average:
            self.logger.info(f"Loading data average")
            self.data = self._get_data_avg()
        else:
            self.logger.info(f"Loading realisation {realisation}")
            self.data = self.pks_all[realisation]

        winfit_file = os.path.abspath(self.data_location + f"/bin0_winfit_{step_size}.txt")
        self._load_winfit(winfit_file)

        winpk_file = os.path.abspath(self.data_location + "/lwin.txt")
        self._load_winpk_file(winpk_file)

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

    def _load_winfit(self, winfit_file):
        # TODO: Add documentation when I figure out how this works
        self.logger.debug(f"Loading winfit from {winfit_file}")
        matrix = np.genfromtxt(winfit_file, skip_header=4)
        self.w_ks_input = matrix[:, 0]
        self.w_k0_scale = matrix[:, 1]
        self.w_transform = matrix[:, 2:]/(np.sum(matrix[:, 2:], axis=0))

        # God I am sorry for doing this manually but the file format is... tricky
        with open(winfit_file, "r") as f:
            self.w_ks_output = np.array([float(x) for x in f.readlines()[2].split()[1:]])

        # Create a mask used from moving from w_pk to the data k values.
        # This is because we can truncate the data start and end values, but we
        # need to generate the model over a wider range of k
        self.w_mask = np.array([np.isclose(x, self.ks).any() for x in self.w_ks_output])
        self.logger.info(f"Winfit matrix has shape {self.w_transform.shape}")

    def _load_winpk_file(self, winpk_file):
        self.logger.debug(f"Loading winpk from {winpk_file}")

        # data files contain (index, k, pk, nk)
        data = np.genfromtxt(winpk_file)
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
    def __init__(self, name="SDSS MGS DR7", average=True, realisation=0, apply_hartlap_correction=False, fake_diag=False, recon=False, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=5, data_dir="sdss_mgs_mocks", postprocess=None):
        super().__init__(min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, name=name, postprocess=postprocess, data_dir=data_dir, average=average, realisation=realisation, apply_hartlap_correction=apply_hartlap_correction, fake_diag=fake_diag)


class MockTaipanPowerSpectrum(MockPowerSpectrum):
    def __init__(self, average=True, realisation=0, min_k=0.02, max_k=0.30, step_size=2, recon=True,
                 reduce_cov_factor=1, name="MockPowerSpectrum", postprocess=None, apply_hartlap_correction=False,
                 fake_diag=False, data_dir="taipan_mocks"):
        super().__init__(min_k=min_k, max_k=max_k, step_size=step_size, recon=recon, reduce_cov_factor=reduce_cov_factor, name=name, postprocess=postprocess, data_dir=data_dir, average=average, realisation=realisation, apply_hartlap_correction=apply_hartlap_correction, fake_diag=fake_diag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    # Some basic checks for data we expect to be there
    dataset = MockPowerSpectrum(step_size=20, recon=False, data_dir="sdss_mgs_mocks")
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
