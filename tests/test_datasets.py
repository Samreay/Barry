from barry.datasets.dataset import Dataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum
from barry.datasets.dataset_correlation_function_abc import CorrelationFunction
from tests.utils import get_concrete


class TestDataset:
    classes = get_concrete(Dataset)
    concrete = []

    @classmethod
    def setup_class(cls):
        cls.concrete = [c() for c in cls.classes]
        print(cls.concrete)

    def test_all_datasets_define_cosmology(self):
        for c in self.concrete:
            datas = c.get_data()
            for data in datas:
                keys = list(data.keys())
                assert "cosmology" in keys, "Data should have a cosmology key!"
                cosmology = data["cosmology"]
                assert isinstance(cosmology, dict)
                required_keys = ["z", "om", "h0", "ns", "ob", "mnu", "reconsmoothscale"]
                for r in required_keys:
                    assert r in cosmology.keys(), f"Cosmology should have key {r}, but has keys {list(cosmology.keys())}"

    def test_all_power_spectrum_datasets(self):
        for c in self.concrete:
            if isinstance(c, PowerSpectrum):
                datas = c.get_data()
                for data in datas:
                    required_keys = ["name", "ks_output", "ks", "pk", "ks_input", "w_scale", "w_transform", "w_pk"]
                    computed_keys = ["w_mask", "num_mocks", "icov", "cov"]
                    for r in required_keys:
                        assert r in data.keys(), f"Power spectrum data needs to have key {r}"
                    for r in computed_keys:
                        assert r in data.keys(), f"Power spectrum data should have computed key {r}"

    def test_all_correlation_function_datasets(self):
        for c in self.concrete:
            if isinstance(c, CorrelationFunction):
                datas = c.get_data()
                for data in datas:
                    required_keys = ["name", "dist", "xi0"]  # "xi", "xi2", "xi4" would be good to have in the future
                    computed_keys = ["num_mocks", "icov", "cov"]
                    for r in required_keys:
                        assert r in data.keys(), f"Power spectrum data needs to have key {r}"
                    for r in computed_keys:
                        assert r in data.keys(), f"Power spectrum data should have computed key {r}"
