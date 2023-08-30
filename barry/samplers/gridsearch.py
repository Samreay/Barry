import logging
import os
import numpy as np
from scipy.optimize import differential_evolution
from barry.samplers.sampler import Sampler


class GridSearch(Sampler):
    def __init__(self, temp_dir=None, ngrid_alpha=101, ngrid_epsilon=101, tol=1.0e-6):

        self.logger = logging.getLogger("barry")
        self.ngrid_alpha = ngrid_alpha
        self.ngrid_epsilon = ngrid_epsilon
        self.temp_dir = temp_dir
        self.tol = tol
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

    def get_file_suffix(self):
        return "gridsearch_chain.npy"

    def fit(self, model, save_dims=None, uid=None):
        """Just runs a simple grid search and stores the grid points in the chain file.

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
        num_dim = model.get_num_dim()
        prior_transform = model.unscale

        filename = self.get_filename(uid)
        if os.path.exists(filename):
            self.logger.info("Not sampling, returning result from GridSearch file.")
            return self.load_file(filename)
        self.logger.info("Sampling posterior now")

        self.logger.debug("Fitting framework with %d dimensions" % num_dim)
        self.logger.info("Using GridSearch")

        if save_dims is None:
            save_dims = num_dim

        # As we are performing a grid-search we need to do some trickery to stop the model treating alpha (and epsilon)
        # as free parameters. So we add them to the list of fixed params and update their default values for every point in the grid.
        num_dim = num_dim - 1
        model.fix_params.extend(["alpha"])
        alpha_grid = np.linspace(model.get_param("alpha").min, model.get_param("alpha").max, self.ngrid_alpha)
        if model.isotropic:
            self.ngrid_epsilon = 1
        else:
            num_dim = num_dim - 1
            model.fix_params.extend(["epsilon"])
            epsilon_grid = np.linspace(model.get_param("epsilon").min, model.get_param("epsilon").max, self.ngrid_epsilon)
        model.set_fix_params(model.fix_params)

        index_val = 1 if model.isotropic else 2
        chain = np.empty((self.ngrid_alpha * self.ngrid_epsilon, index_val + num_dim))
        logp, chi2 = np.empty(self.ngrid_alpha * self.ngrid_epsilon), np.empty(self.ngrid_alpha * self.ngrid_epsilon)
        for i in range(self.ngrid_alpha):
            for j in range(self.ngrid_epsilon):
                chain[i * self.ngrid_epsilon + j, 0] = alpha_grid[i]
                model.set_default("alpha", alpha_grid[i])
                if not model.isotropic:
                    chain[i * self.ngrid_epsilon + j, 1] = epsilon_grid[j]
                    model.set_default("epsilon", epsilon_grid[j])

                # Are there any free parameters left? If not, then we just evaluate the posterior
                if num_dim == 0:
                    logp[i * self.ngrid_epsilon + j] = log_posterior([])
                else:
                    # Otherwise we need to use differential evolution to find the best fit for these values of alpha/epsilon.
                    bounds = [(0.0, 1.0) for _ in range(num_dim)]
                    res = differential_evolution(lambda *x: -log_posterior(prior_transform(*x)), bounds, tol=self.tol)
                    chain[i * self.ngrid_epsilon + j, index_val:] = prior_transform(res.x)
                    logp[i * self.ngrid_epsilon + j] = -res.fun

                chi2[i * self.ngrid_epsilon + j] = model.get_model_summary(model.get_param_dict(chain[i * self.ngrid_epsilon + j]))[0]

        self._save(chain, logp, chi2, filename, save_dims)

        # Now untrick the model so that anything we want to do after sampling works as expected
        num_dim += num_dim + 1
        model.fix_params = [p for p in model.fix_params if p != "alpha"]
        if not model.isotropic:
            num_dim += num_dim + 1
            model.fix_params = [p for p in model.fix_params if p != "epsilon"]
        model.set_fix_params(model.fix_params)

        return {"chain": chain, "posterior": logp}

    def _save(self, chain, likelihood, chi2, filename, save_dims):
        res = np.vstack((likelihood, chi2, chain[:, :save_dims].T)).T
        np.save(filename, res.astype(np.float32))

    def load_file(self, filename):
        """Load existing results from a file"""

        results = np.load(filename)
        likelihood = results[:, 0]
        chi2 = results[:, 1]
        flat_chain = results[:, 2:]
        print(np.shape(flat_chain), np.shape(likelihood), np.shape(chi2))
        return {"chain": np.array(flat_chain), "posterior": np.array(likelihood), "chi2": np.array(chi2)}
