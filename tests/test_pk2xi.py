from barry.cosmology import PowerToCorrelationGauss, PowerToCorrelationFT, getCambGenerator
import numpy as np


class TestPk2Xi:
    camb = None
    ks = None
    pk = None
    truth = None
    ss = np.linspace(20, 200, 100)
    xi_truth = None
    gauss = None
    fft = None
    threshold = 0.3  # This number needs better justification

    @classmethod
    def setup_class(cls):
        cls.camb = getCambGenerator()
        cls.ks = cls.camb.ks
        cls.pk = cls.camb.get_data()["ks"]
        cls.truth = PowerToCorrelationGauss(cls.ks, interpolateDetail=20, a=0.1)
        cls.xi_truth = cls.truth(cls.camb.ks, cls.pk, cls.ss)
        cls.gauss = PowerToCorrelationGauss(cls.ks)
        cls.fft = PowerToCorrelationFT()

    def test_gaussian(self):
        ss2_xi = self.ss ** 2 * self.gauss(self.ks, self.pk, self.ss)
        diff = np.abs(ss2_xi - self.ss ** 2 * self.xi_truth)
        assert np.all(diff < self.threshold)

    def test_fft(self):
        ss2_xi = self.ss ** 2 * self.fft(self.ks, self.pk, self.ss)
        diff = np.abs(ss2_xi - self.ss ** 2 * self.xi_truth)
        assert np.all(diff < self.threshold)
