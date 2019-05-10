import logging
import numpy as np

from barry.framework.cosmology.bao_extractor import extract_bao
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.models.bao_power import PowerSpectrumFit


class BAOExtractor(PowerSpectrumFit):

    def __init__(self, r_s, fit_omega_m=False, smooth_type="hinton2017", name="BAO Extractor Power Spectrum Polynomial Fit"):
        super().__init__(fit_omega_m=fit_omega_m, smooth_type=smooth_type, name=name)
        self.r_s_fiducial = r_s

    def get_model(self, data, p):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], p)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)

        _, pk_model = extract_bao(data["ks_output"], pk_windowed, self.r_s_fiducial)
        pk_final = pk_model[mask]
        return pk_final


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")

    c = CambGenerator()
    r_s, _ = c.get_data()
    model = BAOExtractor(r_s, fit_omega_m=True)

    from barry.framework.datasets.mock_bao_extractor import MockBAOExtractorPowerSpectrum
    dataset = MockBAOExtractorPowerSpectrum(r_s, step_size=2)
    data = dataset.get_data()
    model.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 10}

    import timeit
    n = 100

    def test():
        model.get_likelihood(p)
    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k', label="Data")
        plt.plot(ks, pk2, '-', c='r', label="Model")
        plt.legend()
        plt.show()
