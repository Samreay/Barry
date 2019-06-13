import os
import numpy as np
import logging
import inspect

from barry.framework.dataset import Dataset


class MockAverageCorrelations(Dataset):
    def __init__(self, min_dist=30, max_dist=200, step_size=2, recon=True, reduce_cov_factor=1, name="MockAverageXi"):
        super().__init__(name)
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + "/../../data/taipan_mocks/mock_average/")
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.step_size = step_size
        self.recon = recon

        fs = ["ave", "cov"]
        self.data_filenames = [os.path.abspath(self.data_location + f"/Mock_taipan_year1_v1.xi_{step_size}_{f}_{min_dist}-{max_dist}{'_recon' if recon else ''}") for f in fs]
        for f in self.data_filenames:
            assert os.path.exists(f), f"Cannot find {f}"

        self.logger.info(f"Loading data from {self.data_filenames[0]}")
        self.data = np.loadtxt(self.data_filenames[0])
        self.logger.info(f"Loaded data {self.data.shape}")
        self.logger.info(f"Loading cov from {self.data_filenames[1]}")
        self.cov = np.loadtxt(self.data_filenames[1])
        self.logger.info(f"Loaded cov {self.cov.shape}")
        if reduce_cov_factor != 1:
            self.logger.info(f"Reducing covariance by factor of {reduce_cov_factor}")
            self.cov /= reduce_cov_factor
        self.icov = np.linalg.inv(self.cov)

    def get_data(self):
        return {
            "dist": self.data[:, 0],
            "xi": self.data[:, 1],
            "cov": self.cov,
            "icov": self.icov,
            "name": self.name
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")

    # Some basic checks for data we expect to be there
    dataset = MockAverageCorrelations()
    data = dataset.get_data()

    import matplotlib.pyplot as plt
    import numpy as np
    plt.errorbar(data["dist"], data["dist"]**2 * data["xi"], yerr=data["dist"]**2 * np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()

    # MockAverageCorrelations(min_dist=50, max_dist=170)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3, recon=False)
    # MockAverageCorrelations(step_size=4, recon=False)
