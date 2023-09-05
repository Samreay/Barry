import logging
import os
import numpy as np
from barry.samplers.sampler import Sampler


class EnsembleSampler(Sampler):
    def __init__(self, num_walkers=None, num_steps=20000, temp_dir=None, autoconverge=True, print_progress=False):
        """Uses ``emcee`` and the `EnsembleSampler
        <http://dan.iel.fm/emcee/current/api/#emcee.EnsembleSampler>`_ to fit the supplied
        model.

        This method sets an emcee run using the ``EnsembleSampler`` and manual
        chain management to allow for low to medium dimensional models. MPI running
        is detected automatically for less hassle, and chain progress is serialised
        to disk automatically for convenience.

        Parameters
        ----------
        num_walkers : int, optional
            The number of walkers to run. If not supplied, it defaults to four times the
            framework dimensionality
        num_steps : int, optional
            The maximum number of steps to run
        temp_dir : str
            If set, specifies a directory in which to save temporary results, like the emcee chain
        autoconverge : bool
            Whether or not to perform automated converge checking and stop early if this is achieved
        """

        self.logger = logging.getLogger("barry")
        self.chain = None
        self.pool = None
        self.master = True
        self.num_steps = num_steps
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.num_walkers = num_walkers
        self.autoconverge = autoconverge
        self.print_progress = print_progress

    def get_file_suffix(self):
        return "emcee_chain.npy"

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
            A dictionary with key "chains" containing the final
            flattened chain of dimensions
             ``(num_dimensions, num_walkers * (num_steps - num_burn))``
        """
        import emcee

        log_posterior = model.get_posterior
        start = model.get_start
        num_dim = model.get_num_dim()

        assert log_posterior is not None
        assert start is not None

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info(f"Not sampling, returning result from Emcee file {filename}.")
            return self.load_file(filename)

        if self.num_walkers is None:
            self.num_walkers = num_dim * 4

        if save_dims is None:
            save_dims = num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using Ensemble Sampler")

        pos = start(num_walkers=self.num_walkers)
        self.logger.info("Sampling posterior now")

        sampler = emcee.EnsembleSampler(self.num_walkers, num_dim, log_posterior)

        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        index = 0
        counter = 0
        old_tau = np.inf
        autocorr = np.empty(self.num_steps)
        for sample in sampler.sample(pos, iterations=self.num_steps, progress=self.print_progress):

            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            counter += 100

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            index += 1

        tau = sampler.get_autocorr_time(tol=0)
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
