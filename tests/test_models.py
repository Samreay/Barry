from barry.datasets.dummy import DummyCorrelationFunction_SDSS_DR12_Z061_NGC, DummyPowerSpectrum_SDSS_DR12_Z061_NGC
from barry.models.model import Model
from barry.models.bao_power import PowerSpectrumFit
from barry.models.bao_correlation import CorrelationFunctionFit

from tests.utils import get_concrete
import numpy as np


class TestDataset:
    classes = get_concrete(Model)
    concrete = []
    num_start = 100

    @classmethod
    def setup_class(cls):
        cls.pk_data = DummyPowerSpectrum_SDSS_DR12_Z061_NGC()
        cls.xi_data = DummyCorrelationFunction_SDSS_DR12_Z061_NGC()
        for c in cls.classes:
            model = c()
            if isinstance(model, PowerSpectrumFit):
                model.set_data(cls.pk_data.get_data())
            elif isinstance(model, CorrelationFunctionFit):
                model.set_data(cls.xi_data.get_data())
            cls.concrete.append(model)

    def test_pk_nonnan_likelihood_with_default_param_values(self):
        for c in self.concrete:
            if isinstance(c, PowerSpectrumFit):
                params = c.get_defaults()
                posterior = c.get_posterior(params)
                assert np.isfinite(posterior), f"Model {str(c)} at params {params} gave posterior {posterior}"

    def test_pk_random_starting_point_doesnt_fail(self):
        for c in self.concrete:
            if isinstance(c, PowerSpectrumFit):
                np.random.seed(0)
                for i in range(self.num_start):
                    params = c.get_raw_start()
                    posterior = c.get_posterior(params)
                    assert np.isfinite(posterior), f"Model {str(c)} at params {params} gave posterior {posterior}"

    def test_xi_nonnan_likelihood_with_default_param_values(self):
        for c in self.concrete:
            if isinstance(c, CorrelationFunctionFit):
                params = c.get_defaults()
                posterior = c.get_posterior(params)
                assert np.isfinite(posterior), f"Model {str(c)} at params {params} gave posterior {posterior}"

    def test_xi_random_starting_point_doesnt_fail(self):
        for c in self.concrete:
            if isinstance(c, CorrelationFunctionFit):
                np.random.seed(0)
                for i in range(self.num_start):
                    params = c.get_raw_start()
                    posterior = c.get_posterior(params)
                    assert np.isfinite(posterior), f"Model {str(c)} at params {params} gave posterior {posterior}"
