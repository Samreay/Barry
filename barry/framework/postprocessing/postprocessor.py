import logging
from abc import abstractmethod, ABC


class PostProcess(ABC):
    def __init__(self):
        self.logger = logging.getLogger("barry")

    @abstractmethod
    def __call__(self, **inputs):
        pass


class PkPostProcess(PostProcess):
    def __call__(self, **inputs):
        return self.postprocess(inputs["ks"], inputs["pk"])

    @abstractmethod
    def postprocess(self, ks, pk):
        pass
