import logging
import os
import numpy as np
from scipy.optimize import differential_evolution
from barry.samplers.sampler import Sampler


class Optimiser(Sampler):
    def __init__(self, temp_dir=None, tol=1.0e-6):

        self.logger = logging.getLogger("barry")
        self.tol = tol
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

    def get_file_suffix(self):
        return "bestfit_chain.npy"

    def fit(self, model, save_dims=None, uid=None):
        """ " Just runs a simple optimisation and stores the best fit in the chain file.

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
                - *chain*: the best fit point
                - *posterior*: the likelihood at this point
        """

        log_posterior = model.get_posterior
        start = model.get_start
        num_dim = model.get_num_dim()
        prior_transform = model.unscale

        assert log_posterior is not None
        assert start is not None
        assert prior_transform is not None

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info(f"Not sampling, returning result from Bestfit file {filename}.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using Optimiser")

        bounds = [(0.0, 1.0) for _ in range(num_dim)]
        res = differential_evolution(lambda *x: -log_posterior(prior_transform(*x)), bounds, tol=self.tol)

        ps = prior_transform(res.x)
        print(res.fun, ps)
        np.save(filename, np.concatenate([[-res.fun], ps]))

        return {"chain": ps, "posterior": -res.fun}

    def load_file(self, filename):
        """Load existing results from a file"""

        results = np.load(filename)
        likelihood = [results[0]]
        flat_chain = results[1:][None, :]
        return {"chain": np.array(flat_chain), "posterior": np.array(likelihood)}
