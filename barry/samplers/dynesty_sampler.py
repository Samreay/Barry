import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class DynestySampler(Sampler):
    def __init__(self, temp_dir=None, max_iter=None, dynamic=False, nlive=500, print_progress=False):

        self.logger = logging.getLogger("barry")
        self.max_iter = max_iter
        self.nlive = nlive
        # dynesty.utils.merge_runs()
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.dynamic = dynamic
        self.print_progress = print_progress

    def get_filename(self, uid):
        if self.dynamic:
            return os.path.join(self.temp_dir, f"{uid}_dyn_chain.npy")
        else:
            return os.path.join(self.temp_dir, f"{uid}_nest_chain.npy")

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
        import dynesty

        log_likelihood = model.get_posterior
        num_dim = model.get_num_dim()
        prior_transform = model.unscale

        assert log_likelihood is not None
        assert prior_transform is not None

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info("Not sampling, returning result from file.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using dynesty Sampler")
        if self.dynamic:
            sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, num_dim)
            sampler.run_nested(
                maxiter=self.max_iter, print_progress=self.print_progress, nlive_init=self.nlive, nlive_batch=100, maxbatch=10
            )
        else:
            sampler = dynesty.NestedSampler(log_likelihood, prior_transform, num_dim, nlive=self.nlive)
            sampler.run_nested(maxiter=self.max_iter, print_progress=self.print_progress)

        self.logger.debug("Fit finished")

        dresults = sampler.results
        logz = dresults["logz"]
        chain = dresults["samples"]
        weights = np.exp(dresults["logwt"] - dresults["logz"][-1])
        max_weight = weights.max()
        trim = max_weight / 1e5
        mask = weights > trim
        likelihood = dresults["logl"]
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
