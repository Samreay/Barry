import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class NautilusSampler(Sampler):
    def __init__(self, temp_dir=None, max_iter=None, nlive=3000, nbatch = 1000, print_progress=False, vectorized = False):

        self.logger = logging.getLogger("nautilus")
        self.max_iter = max_iter
        self.nlive = nlive
        self.nbatch = nbatch
        self.vectorized = vectorized
        if self.vectorized:
            raise ValueError("Barry likelihoods do not yet support vectorization.")
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.print_progress = print_progress

    def get_filename(self, uid):
        return os.path.join(self.temp_dir, f"{uid}_nest_chain.npy")

    def fit(self, log_likelihood, start, num_dim, prior_transform, save_dims=None, uid=None):

        import nautilus

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info("Not sampling, returning result from file.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using nautilus Sampler")
        sampler = nautilus.Sampler(prior_transform, log_likelihood, 
                                   n_dim = num_dim, 
                                   n_live = self.nlive, 
                                   n_batch = self.nbatch, 
                                   vectorized = self.vectorized, 
                                   pass_dict = False, 
                                   pool = None, 
                                   filepath = f"{self.temp_dir}/results.h5", 
                                   resume = True)
        dresults = sampler.run(verbose = self.print_progress)

        self.logger.debug("Fit finished")

        points, log_w, log_l = sampler.posterior()
        cumsumweights = np.cumsum(np.exp(log_w))
        mask = cumsumweights > 1e-4
        chain = points
        logz = sampler.evidence()
        weights = np.exp(log_w)
        max_weight = weights.max()
        trim = max_weight / 1e5
        mask = weights > trim
        likelihood = log_l
        self._save(chain[mask, :], weights[mask], likelihood[mask], filename, logz[mask], save_dims)
        return {"chain": chain[mask, :], "weights": weights[mask], "posterior": likelihood[mask], "evidence": logz}

    def _save(self, chain, weights, likelihood, filename, logz, save_dims):
        res = np.vstack((likelihood, weights, logz, chain[:, :save_dims].T)).T
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        results = np.load(filename)
        likelihood = results[:, 0]
        weights = results[:, 1]
        logz = results[:, 2]
        flat_chain = results[:, 3:]
        return {"chain": flat_chain, "posterior": likelihood, "evidence": logz, "weights": weights}
