import logging

from barry.datasets.dataset_correlation_function_abc import CorrelationFunction


class CorrelationFunction_SDSS_DR7_Z015_MGS(CorrelationFunction):
    def __init__(self, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        super().__init__("sdss_dr7_corr.pkl", min_dist, max_dist, recon, reduce_cov_factor, realisation)


class CorrelationFunction_SDSS_DR12_Z061_NGC(CorrelationFunction):
    def __init__(self, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None):
        super().__init__("sdss_dr12_ngc_corr_zbin0p61.pkl", min_dist, max_dist, recon, reduce_cov_factor, realisation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Some basic checks for data we expect to be there
    dataset = CorrelationFunction_SDSS_DR7_Z015_MGS()
    data = dataset.get_data()

    import matplotlib.pyplot as plt
    import numpy as np

    plt.errorbar(data["dist"], data["dist"] ** 2 * data["xi0"], yerr=data["dist"] ** 2 * np.sqrt(np.diag(data["cov"])), fmt="o", c="k")
    plt.show()

    # MockAverageCorrelations(min_dist=50, max_dist=170)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3)
    # MockAverageCorrelations(min_dist=50, max_dist=170, step_size=3, recon=False)
    # MockAverageCorrelations(step_size=4, recon=False)
