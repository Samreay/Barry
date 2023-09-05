import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class ZeusSampler(Sampler):
    def __init__(self, num_walkers=None, temp_dir=None, num_steps=20000, autoconverge=True, print_progress=False):

        self.logger = logging.getLogger("barry")
        self.num_steps = num_steps
        self.num_walkers = num_walkers
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.autoconverge = autoconverge
        self.print_progress = print_progress

    def get_file_suffix(self):
        return "zeus_chain.npy"

    def fit(self, model, save_dims=None, uid=None):
        """
        Fit the model

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
            A dictionary containing the chain and the weights
        """
        import zeus

        log_posterior = model.get_posterior
        start = model.get_start
        num_dim = model.get_num_dim()

        assert log_posterior is not None
        assert start is not None

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info(f"Not sampling, returning result from Zeus file {filename}.")
            return self.load_file(filename)

        if self.num_walkers is None:
            self.num_walkers = num_dim * 4

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)

        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using Zeus Sampler")

        callbacks = []
        if self.autoconverge:
            # Default convergence criteria from Zeus docos. Seem reasonable.
            cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50)
            cb1 = zeus.callbacks.MinIterCallback(nmin=500)
            callbacks = [cb0, cb1]

        pos = start(num_walkers=self.num_walkers)
        self.logger.info("Sampling posterior now")

        sampler = zeus.EnsembleSampler(self.num_walkers, num_dim, log_posterior, verbose=False)
        sampler.run_mcmc(pos, self.num_steps, callbacks=callbacks, progress=self.print_progress)

        self.logger.debug("Fit finished")

        tau = zeus.AutoCorrTime(sampler.get_chain())
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True)
        likelihood = sampler.get_log_prob(discard=burnin, flat=True)
        self._save(samples, likelihood, filename, save_dims)
        return {"chain": samples, "weights": np.ones(len(likelihood)), "posterior": likelihood}

    def _save(self, chain, likelihood, filename, save_dims):
        res = np.vstack((likelihood, chain[:, :save_dims].T)).T
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        results = np.load(filename)
        likelihood = results[:, 0]
        flat_chain = results[:, 1:]
        return {"chain": flat_chain, "weights": np.ones(len(likelihood)), "posterior": likelihood}
