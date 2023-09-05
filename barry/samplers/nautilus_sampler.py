import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class NautilusSampler(Sampler):
    def __init__(self, temp_dir=None, max_iter=None, dynamic=False, nlive=500, nupdate=None, print_progress=False):

        self.logger = logging.getLogger("barry")
        self.max_iter = max_iter
        self.nlive = nlive
        self.nupdate = nupdate
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.dynamic = dynamic
        self.print_progress = print_progress

    def get_file_suffix(self):
        return "nautilus_chain.npy"

    def fit(self, model, save_dims=None, uid=None):
        """Runs the sampler over the model and returns the flat chain of results

        Parameters
        ----------
        model : class <Model>
            An instance of one of barry's model classes
        save_dims : int, optional
            Only return values for the first ``save_dims`` parameters.
            Useful to remove numerous marginalisation parameters if running
            low on memory or hard drive space.
        uid : str, optional
            A unique identifier used to differentiate different fits
            if two fits both serialise their chains and use the
            same temporary directory
        Returns
        -------
        dict
            A dictionary of results containing:
                - *chain*: the chain
                - *weights*: chain weights if applicable
                - *posterior*: posterior values for each point in the chain
                - *evidence*: Bayesian evidence for the model/data combo.

        """
        import nautilus

        log_likelihood = model.get_posterior
        num_dim = model.get_num_dim()
        prior_transform = model.unscale

        assert log_likelihood is not None
        assert prior_transform is not None

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info(f"Not sampling, returning result from Nautilus file {filename}.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using Nautilus Sampler")
        sampler = nautilus.Sampler(prior_transform, log_likelihood, n_dim=num_dim, n_live=self.nlive, n_update=self.nupdate)
        sampler.run(verbose=self.print_progress, discard_exploration=True)

        self.logger.debug("Fit finished")

        chain, logw, likelihood = sampler.posterior()
        logz = sampler.evidence()
        weights = np.exp(logw)
        max_weight = weights.max()
        trim = max_weight / 1e5
        mask = weights > trim
        self._save(chain[mask, :], weights[mask], likelihood[mask], filename, np.zeros(len(mask))[mask] + logz, save_dims)
        return {
            "chain": chain[mask, :],
            "weights": weights[mask],
            "posterior": likelihood[mask],
            "evidence": np.zeros(len(mask))[mask] + logz,
        }

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
