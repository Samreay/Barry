import logging
from abc import abstractmethod, ABC


class PostProcess(ABC):
    def __init__(self):
        self.logger = logging.getLogger("barry")

    @abstractmethod
    def __call__(self, **inputs):
        pass


class PkPostProcess(PostProcess):
    """ An abstract implementation of PostProcess for power spectrum models.

    Requires that implementations pass in k values, p(k) values and a boolean mask.
    """
    def __call__(self, **inputs):
        return self.postprocess(inputs["ks"], inputs["pk"], inputs["mask"])

    @abstractmethod
    def postprocess(self, ks, pk, mask):
        pass


class XiPostProcess(PostProcess):
    """ An abstract implementation of PostProcess for correlation function models.

    Requires that implementations pass in dist values, xi(s) values and a boolean mask.

    Note that xi(s) is assumed to be the xi_0 aka the monopole.

    As the BAO extractor is a Pk model only this class is not used... but it could be.
    """
    def __call__(self, **inputs):
        return self.postprocess(inputs["dist"], inputs["xi"], inputs["mask"])

    @abstractmethod
    def postprocess(self, dist, xi, mask):
        pass
