import logging
import os
import numpy as np
from barry.samplers.hdemcee import EmceeWrapper
from barry.samplers.sampler import Sampler


class EnsembleSampler(Sampler):
    def __init__(self, num_walkers=None, num_steps=1000, num_burn=300, temp_dir=None, save_interval=300):
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
            The number of walkers to run. If not supplied, it defaults to eight times the
            framework dimensionality
        num_steps : int, optional
            The number of steps to run
        num_burn : int, optional
            The number of steps to discard for burn in
        temp_dir : str
            If set, specifies a directory in which to save temporary results, like the emcee chain
        save_interval : float
            The amount of seconds between saving the chain to file. Setting to ``None``
            disables serialisation.
        """

        self.logger = logging.getLogger("barry")
        self.chain = None
        self.pool = None
        self.master = True
        self.num_steps = num_steps
        self.num_burn = num_burn
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.save_interval = save_interval
        self.num_walkers = num_walkers

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

        if self.num_walkers is None:
            self.num_walkers = num_dim * 8
            self.num_walkers = max(self.num_walkers, 50)

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)

        self.logger.info("Using Ensemble Sampler")
        sampler = emcee.EnsembleSampler(self.num_walkers, num_dim, log_posterior, live_dangerously=True)

        emcee_wrapper = EmceeWrapper(sampler)
        flat_chain = emcee_wrapper.run_chain(
            self.num_steps,
            self.num_burn,
            self.num_walkers,
            num_dim,
            start=start,
            save_dim=save_dims,
            temp_dir=self.temp_dir,
            uid=uid,
            save_interval=self.save_interval,
        )
        self.logger.debug("Fit finished")
        if self.pool is not None:  # pragma: no cover
            self.pool.close()
            self.logger.debug("Pool closed")
        return {"chain": flat_chain, "weights": np.ones(flat_chain.shape[0])}

    def load_file(self, filename):
        results = np.load(filename)
        posterior = np.load(filename.replace("chain.npy", "prob.npy"))
        flat_chain = results[:, self.num_burn :, :].reshape((-1, results.shape[2]))
        flat_posterior = posterior[:, self.num_burn :].reshape((-1, 1))
        return {"chain": flat_chain, "posterior": flat_posterior}
