import logging
import os
import numpy as np
from barry.framework.samplers.sampler import GenericSampler


class DynestySampler(GenericSampler):

    def __init__(self, temp_dir=None, max_iter=None, nlive=100):

        self.logger = logging.getLogger("barry")
        self.max_iter = max_iter
        self.nlive = nlive
        import dynesty
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

    def get_filename(self, uid):
        return os.path.join(self.temp_dir, f"{uid}_chain.npy")


    def fit(self, log_likelihood, start, prior_transform, save_dims=None, uid=None):

        import dynesty

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info("Not sampling, returning result from file.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        if callable(start):
            num_dim = np.array(start()).size
        else:
            num_dim = np.array(start.size)
        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using dynesty Sampler")
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, num_dim, nlive=self.nlive)

        sampler.run_nested(maxiter=self.max_iter, print_progress=False)

        self.logger.debug("Fit finished")

        dresults = sampler.results
        chain = dresults["samples"]
        weights = np.exp(dresults['logwt'] - dresults['logz'][-1])
        likelihood = dresults["logl"]
        self._save(chain, weights, likelihood, filename, save_dims)
        return {"chain": chain, "weights": weights, "posterior": likelihood}

    def _save(self, chain, weights, likelihood, filename, save_dims):
        res = np.vstack((likelihood, weights, chain[:, :save_dims].T)).T
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        results = np.load(filename)
        likelihood = results[:, 0]
        weights = results[:, 1]
        flat_chain = results[:, 2:]
        return {"chain": flat_chain, "posterior": likelihood, "weights": weights}
