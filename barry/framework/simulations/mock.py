from barry.framework.simulation import Simulation


class Mock(Simulation):
    def __init__(self):
        super().__init__("Mock")

    def get_data(self):
        return None
