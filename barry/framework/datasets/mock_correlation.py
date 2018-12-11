import os
import numpy as np
import logging
import inspect

from barry.framework.dataset import Dataset


class MockCorrelations(Dataset):
    def __init__(self, step_size=2, realisation=0):
        super().__init__("MockAverage")
        self.data_location = "../../data/taipan_mocks/mock_individual/"
        self.step_size = step_size
        self.realisation = realisation
        return
        # TODO: finish this after clarifying with Cullan
        fs = ["ave", "cov"]
        self.data_filenames = [os.path.abspath(self.data_location + f"Mock_taipan_year1_v1.xi_{step_size}_{f}_{min_dist}-{max_dist}{'_recon' if recon else ''}") for f in fs]
        for f in self.data_filenames:
            assert os.path.exists(f), f"Cannot find {f}"

        self.logger.info(f"Loading data from {self.data_filenames[0]}")
        self.data = np.loadtxt(self.data_filenames[0])
        self.logger.info(f"Loaded data {self.data.shape}")
        self.logger.info(f"Loading cov from {self.data_filenames[1]}")
        self.cov = np.loadtxt(self.data_filenames[1])
        self.logger.info(f"Loaded cov {self.cov.shape}")

    def get_data(self):
        return self.data, self.cov


class MockAverageCorrelations(Dataset):
    def __init__(self, min_dist=30, max_dist=200, step_size=2, recon=True):
        super().__init__("MockAverage")
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

    def get_data(self):
        return self.data, self.cov

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")

    # Some basic checks for data we expect to be there
    MockAverageCorrelations()
    MockAverageCorrelations(min_dist=50, max_dist=170)
    MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3)
    MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3, recon=False)
    MockAverageCorrelations(step_size=4, recon=False)
