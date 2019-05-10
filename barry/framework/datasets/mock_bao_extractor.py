import logging
from barry.framework.cosmology.bao_extractor import extract_bao
from barry.framework.datasets.mock_power import MockPowerSpectrum


class MockBAOExtractorPowerSpectrum(MockPowerSpectrum):
    def __init__(self, r_s, delta=0.5, average=True, realisation=0, min_k=0.02, max_k=0.30, step_size=2, recon=True,
                 reduce_cov_factor=1, name="BAOExtractor"):
        self.r_s = r_s
        self.delta = delta
        super().__init__(average, realisation, min_k, max_k, step_size, recon, reduce_cov_factor, name)

    def _rebin_data(self, dataframe):
        k_rebinned, pk_rebinned, mask = self._agg_data(dataframe)
        _, p_extracted = extract_bao(k_rebinned, pk_rebinned, r_s=self.r_s, delta=self.delta)
        return k_rebinned[mask], p_extracted[mask]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")
    from barry.framework.cosmology.camb_generator import CambGenerator

    # Some basic checks for data we expect to be there
    c = CambGenerator()
    r_s, _ = c.get_data()
    dataset = MockBAOExtractorPowerSpectrum(r_s, step_size=2)
    data = dataset.get_data()
    print(data["ks"])

    import matplotlib.pyplot as plt
    import numpy as np
    plt.errorbar(data["ks"], data["pk"], yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=1)
    # MockAveragePowerSpectrum(min_k=0.02, max_k=0.30, step_size=2, recon=False)
    # MockAveragePowerSpectrum(step_size=5, recon=False)
