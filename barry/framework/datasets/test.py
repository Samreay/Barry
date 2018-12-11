from barry.framework.dataset import Dataset
from scipy.stats import norm


class TestDataset(Dataset):
    def __init__(self):
        super().__init__("TestDataset")
        self.data = norm.rvs(loc=0.3, scale=1.0, size=2000)

    def get_data(self):
        return self.data
