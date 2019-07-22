import os
import numpy as np
import pickle
import logging
import inspect

from barry.framework.dataset import Dataset


class MockAverageCorrelations(Dataset):
    def __init__(self, filename, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        current_file = os.path.dirname(inspect.stack()[0][1])
        self.data_location = os.path.normpath(current_file + f"/../../data/{filename}")
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.recon = recon
        self.reduce_cov_factor = reduce_cov_factor

        with open(self.data_location, "rb") as f:
            self.data_obj = pickle.load(f)
        super().__init__(self.data_obj["name"])
        self.cosmology = self.data_obj["cosmology"]

        self.all_data = self.data_obj["post-recon"] if recon else self.data_obj["pre-recon"]
        self.cov, self.icov, self.data = None, None, None
        self.set_realisation(realisation)
        self._compute_cov()

    def set_realisation(self, realisation):
        if realisation is None:
            self.data = np.array(self.all_data).mean(axis=0)
        else:
            self.data = self.all_data[realisation]

    def _compute_cov(self):
        # TODO: Generalise for other multipoles poles
        x0 = np.array(self.all_data)[:, :, 2]
        cov = np.cov(x0.T)
        self.cov = cov / self.reduce_cov_factor
        self.icov = np.linalg.inv(self.cov)

    def get_data(self):
        return {
            "dist": self.data[:, 0],
            "xi": self.data[:, 1],
            "xi0": self.data[:, 2],
            "xi2": self.data[:, 3],
            "xi4": self.data[:, 4],
            "cov": self.cov,
            "icov": self.icov,
            "name": self.name
        }


class MockSDSSCorrelationFunction(MockAverageCorrelations):
    def __init__(self, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        super().__init__("sdss_dr7_corr.pkl", min_dist, max_dist, recon, reduce_cov_factor, realisation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Some basic checks for data we expect to be there
    dataset = MockSDSSCorrelationFunction()
    data = dataset.get_data()

    import matplotlib.pyplot as plt
    import numpy as np
    plt.errorbar(data["dist"], data["dist"]**2 * data["xi0"], yerr=data["dist"]**2 * np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()

    # MockAverageCorrelations(min_dist=50, max_dist=170)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3, recon=False)
    # MockAverageCorrelations(step_size=4, recon=False)
