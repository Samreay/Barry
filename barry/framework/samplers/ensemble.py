
import logging
import sys
import os
import numpy as np
from barry.framework.samplers.hdemcee import EmceeWrapper
from barry.framework.samplers.sampler import GenericSampler


class EnsembleSampler(GenericSampler):

    def __init__(self, num_walkers=None, num_steps=5000, num_burn=2000,
                 temp_dir=None, save_interval=300):
        """ Uses ``emcee`` and the `EnsembleSampler
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

        self.logger = logging.getLogger(__name__)
        import emcee
        self.chain = None
        self.pool = None
        self.master = True
        self.num_steps = num_steps
        self.num_burn = num_burn
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        self.save_interval = save_interval
        self.num_walkers = num_walkers

    def fit(self, log_posterior, start, save_dims=None, uid=None):
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
        from emcee.utils import MPIPool
        import emcee
        try:  # pragma: no cover
            self.pool = MPIPool()
            if not self.pool.is_master():
                self.logger.info("Slave waiting")
                self.master = False
                self.pool.wait()
                sys.exit(0)
            else:
                self.logger.info("MPIPool successful initialised and master found. "
                                 "Running with %d cores." % self.pool.size)
        except ImportError:
            self.logger.info("mpi4py is not installed or not configured properly. "
                             "Ignore if running through python, not mpirun")
        except ValueError as e:  # pragma: no cover
            self.logger.info("Unable to start MPI pool, expected normal python execution")
            self.logger.info(str(e))

        if callable(start):
            num_dim = np.array(start()).size
        else:
            num_dim = np.array(start.size)
        if self.num_walkers is None:
            self.num_walkers = num_dim * 4
            self.num_walkers = max(self.num_walkers, 20)

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)

        self.logger.info("Using Ensemble Sampler")
        sampler = emcee.EnsembleSampler(self.num_walkers, num_dim,
                                        log_posterior,
                                        pool=self.pool, live_dangerously=True)

        emcee_wrapper = EmceeWrapper(sampler)
        flat_chain = emcee_wrapper.run_chain(self.num_steps, self.num_burn,
                                             self.num_walkers, num_dim,
                                             start=start,
                                             save_dim=save_dims,
                                             temp_dir=self.temp_dir,
                                             uid=uid,
                                             save_interval=self.save_interval)
        self.logger.debug("Fit finished")
        if self.pool is not None:  # pragma: no cover
            self.pool.close()
            self.logger.debug("Pool closed")
        return {"chain": flat_chain, "weights": np.ones(flat_chain.shape[0])}

    def load_file(self, filename):
        results = np.load(filename)
        flat_chain = results[:, self.num_burn:, :].reshape((-1, results.shape[2]))
        return {"chain": flat_chain}