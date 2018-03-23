from barry.framework.simulation import Simulation
from scipy.stats import norm


class Mock(Simulation):
    def __init__(self):
        super().__init__("Mock")
        self.data = norm.rvs(loc=0.3, scale=1.0, size=1000)

    def get_data(self):
        return self.data
