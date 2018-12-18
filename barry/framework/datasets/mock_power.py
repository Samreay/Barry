import os
import numpy as np
import logging
import inspect

from barry.framework.dataset import Dataset


class MockAveragePowerSpectrum(Dataset):
    def __init__(self, min_k=0.02, max_k=0.30, step_size=2, recon=True, reduce_cov_factor=1, name="MockAveragePK"):
        super().__init__(name)
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + "/../../data/taipan_mocks/mock_average/")
        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.recon = recon

        fs = ["ave", "cov"]
        self.data_filenames = [os.path.abspath(
            self.data_location + f"/Mock_taipan_year1_v1.lpow_{step_size}_0p{int(100 * min_k):02d}-0p{int(100 * max_k)}_{f}{'_recon' if recon else ''}")
                               for f in fs]
        for f in self.data_filenames:
            assert os.path.exists(f), f"Cannot find {f}"

        self.logger.debug(f"Loading data from {self.data_filenames[0]}")
        self.data = np.loadtxt(self.data_filenames[0])
        self.logger.info(f"Loaded data {self.data.shape}")
        self.logger.debug(f"Loading cov from {self.data_filenames[1]}")
        self.cov = np.loadtxt(self.data_filenames[1])
        self.logger.info(f"Loaded cov {self.cov.shape}")
        if reduce_cov_factor != 1:
            self.logger.info(f"Reducing covariance by factor of {reduce_cov_factor}")
            self.cov /= reduce_cov_factor
        self.icov = np.linalg.inv(self.cov)

        winfit_file = os.path.abspath(self.data_location + f"/../taipanmock_year1_mock_rand_cullan.winfit_{step_size}")
        self._load_winfit(winfit_file)

        winpk_file = os.path.abspath(self.data_location + "/../taipanmock_year1_mock_rand_cullan.lwin")
        self._load_winpk_file(winpk_file)

    def _load_winfit(self, winfit_file):
        # TODO: Add documentation when I figure out how this works
        self.logger.debug(f"Loading winfit from {winfit_file}")
        matrix = np.genfromtxt(winfit_file, skip_header=4)
        self.w_ks_input = matrix[:, 0]
        self.w_k0_scale = matrix[:, 1]
        self.w_transform = matrix[:, 2:]

        # God I am sorry for doing this manually but the file format is... tricky
        with open(winfit_file, "r") as f:
            self.w_ks_output = np.array([float(x) for x in f.readlines()[2].split()[1:]])

        # Create a mask used from moving from w_pk to the data k values.
        # This is because we can truncate the data start and end values, but we
        # need to generate the model over a wider range of k
        self.w_mask = np.array([x in self.data[:, 1] for x in self.w_ks_output])
        self.logger.info(f"Winfit matrix has shape {self.w_transform.shape}")

    def _load_winpk_file(self, winpk_file):
        self.logger.debug(f"Loading winpk from {winpk_file}")

        # data files contain (index, k, pk, nk)
        data = np.genfromtxt(winpk_file)
        pk = data[:, 2].reshape((-1, self.step_size))
        weight = data[:, 3].reshape((-1, self.step_size))

        # Take the average of every group of step_size rows to rebin
        self.w_pk = np.average(pk, axis=1, weights=weight)
        self.logger.info(f"Loaded winpk with shape {self.w_pk.shape}")

    def get_data(self):
        return {
            "ks_output": self.w_ks_output,
            "ks": self.data[:, 1],
            "pk": self.data[:, 2],
            "cov": self.cov,
            "icov": self.icov,
            "ks_input": self.w_ks_input,
            "w_scale": self.w_k0_scale,
            "w_transform": self.w_transform,
            "w_pk": self.w_pk,
            "w_mask": self.w_mask
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    # Some basic checks for data we expect to be there
    MockAveragePowerSpectrum(step_size=10)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=1)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=2, recon=False)
    # MockAveragePowerSpectrum(step_size=5, recon=False)
