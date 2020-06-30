import logging
import os
import numpy as np
import emcee
from barry.samplers.sampler import Sampler


class Local_Emcee(Sampler):
    def __init__(self, num_walkers=None, num_steps=40000, temp_dir=None):
        """ Uses ``emcee`` and the `EnsembleSampler
        <http://dan.iel.fm/emcee/current/api/#emcee.EnsembleSampler>`_ to fit the supplied
        model.

        This is a not very clever sampler that is designed to just run on a single processor/laptop
        and run a model till some convergence criterion is reached. Useful for debugging or checking
        runtime till convergence. Burn-in is computed automatically based on convergence criterion.
        Overall, based on example on emcee website.

        Parameters
        ----------
        num_walkers : int, optional
            The number of walkers to run. If not supplied, it defaults to eight times the
            framework dimensionality
        num_steps : int, optional
            The maximum number of steps to run
        temp_dir : str
            If set, specifies a directory in which to save temporary results, like the emcee chain
        save_interval : float
            The amount of seconds between saving the chain to file. Setting to ``None``
            disables serialisation.
        """

        self.logger = logging.getLogger("barry")
        self.chain = None
        self.num_steps = num_steps
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.num_walkers = num_walkers

    def fit(self, log_posterior, start, num_dim, prior_transform, save_dims=None, uid=None):
        """ Runs the sampler over the model and returns the flat chain of results

        Parameters
        ----------
        log_posterior : function
            A function which takes a list of parameters and returns
            the log posterior
        start : function|list|ndarray
            Either a starting position, or a function that can be called
            to generate a starting position
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
        assert log_posterior is not None
        assert start is not None

        if self.num_walkers is None:
            self.num_walkers = num_dim * 8
            self.num_walkers = max(self.num_walkers, 50)

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)

        self.logger.info("Using Local Emcee Ensemble Sampler")

        if save_dims is not None:
            assert save_dims <= num_dim, "You cannot save more dimensions than you actually have"
        else:
            save_dims = num_dim

        past_chain = None
        pos = None
        if self.temp_dir is not None:
            self.logger.debug("Looking in temp dir %s" % self.temp_dir)
            chain_file = os.path.join(self.temp_dir, uid + "_locens.chain.hdf5")
            backend = emcee.backends.HDFBackend(chain_file)
            if os.path.exists(chain_file):
                past_chain = self.read_chain_backend(backend)
                self.logger.info("Found chain of %d steps" % past_chain.shape[1])

        if start is None and pos is None:
            raise ValueError("You need to have either a starting function or existing chains")

        if pos is None:
            pos = start(num_walkers=self.num_walkers)

        sampler = emcee.EnsembleSampler(self.num_walkers, num_dim, log_posterior, backend=backend)

        step = 0
        if past_chain is not None:
            step = backend.iteration
            num = self.num_steps - step
            old_tau = sampler.get_autocorr_time(tol=0)
            self.logger.debug("Further steps (%d) may be required" % num)
        else:
            backend.reset(self.num_walkers, num_dim)
            old_tau = np.inf
            num = self.num_steps
            self.logger.debug("Running full chain of %d steps" % self.num_steps)

        # Run the sampler for a max number iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%.
        counter = 0
        for sample in sampler.sample(pos, iterations=num, progress=True):

            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            counter += 100
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Mean Auto-Correlation time: {0:.3f}".format(np.mean(tau)))

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, np.mean(tau)))
                break
            old_tau = tau

        burnin = int(2 * np.max(tau))
        flat_chain = sampler.get_chain(discard=burnin, flat=True)

        self.logger.debug("Fit finished")
        return {"chain": flat_chain, "weights": np.ones(flat_chain.shape[0])}

    def read_chain_backend(self, backend):

        tau = backend.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        samples = backend.get_chain(discard=burnin, flat=True)
        log_prob_samples = backend.get_log_prob(discard=burnin, flat=True)

        return samples
