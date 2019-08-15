import logging
import os
import numpy as np
from barry.framework.samplers.sampler import GenericSampler


class DynestySampler(GenericSampler):

    def __init__(self, temp_dir=None, max_iter=None):

        self.logger = logging.getLogger("barry")
        self.max_iter = max_iter
        import dynesty
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

    def fit(self, log_likelihood, start, prior_transform, save_dims=None, uid=None):

        import dynesty

        if callable(start):
            num_dim = np.array(start()).size
        else:
            num_dim = np.array(start.size)
        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using dynesty Sampler")
        sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, num_dim)

        sampler.run_nested(maxiter=self.max_iter)

        self.logger.debug("Fit finished")

        dresults = sampler.results
        chain = dresults["samples"]
        weights = np.exp(dresults['logwt'] - dresults['logz'][-1])
        likelihood = dresults["logl"]
        self._save(chain, weights, likelihood, uid, save_dims)
        return {"chain": chain, "weights": weights, "posterior": likelihood}

    def _save(self, chain, weights, likelihood, uid, save_dims):
        res = np.vstack((likelihood, weights, chain[:, :save_dims].T)).T
        filename = os.path.join(self.temp_dir, f"{uid}_chain.npy")
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        results = np.load(filename)
        likelihood = results[:, 0]
        weights = results[:, 1]
        flat_chain = results[:, 2:]
        return {"chain": flat_chain, "posterior": likelihood, "weights": weights}
