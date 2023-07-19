import os
import numpy as np
from time import time
import logging

from barry.samplers.sampler import Sampler


class MetropolisHastings(Sampler):
    """Self tuning Metropolis Hastings Sampler

    Parameters
    ----------
    num_burn : int, optional
        The number of burn in steps. TODO: Tune this automatically
    num_steps : int, optional
        The number of steps to take after burn in. TODO: Tune this automatically
    sigma_adjust : int, optional
        During the burn in, how many steps between adjustment to the step size
    covariance_adjust : int, optional
        During the burn in, how many steps between adjustment to the parameter covariance.
    temp_dir : str, optional
        The location of a folder to save the results, such as the last position and chain
    save_interval : int, optional
        How many seconds should pass between saving data snapshots
    accept_ratio : float, optional
        The desired acceptance ratio
    callback : function, optional
        If set, passes the log posterior, position and weight for each step in the burn
        in and the chain to the function. Useful for plotting the walks whilst the
        chain is running.
    plot_covariance : bool, optional
        If set, plots the covariance matrix to a file in the temp directory.
        As such, requires temp directory to be set.
    num_start : int, optional
        How many starting positions to trial, if the ``start`` value given
        is a function.
    """

    space = 3  # log posterior, sigma, weight
    IND_P = 0
    IND_S = 1
    IND_W = 2

    def __init__(
        self,
        num_burn=3000,
        num_steps=10000,
        sigma_adjust=100,
        covariance_adjust=1000,
        temp_dir=None,
        save_interval=300,
        accept_ratio=0.234,
        callback=None,
        plot_covariance=False,
    ):
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        self.logger = logging.getLogger("barry")
        self.log_posterior = None
        self.start = None
        self.save_dims = None
        self.callback = callback
        self.do_plot_covariance = plot_covariance

        self.num_burn = num_burn
        self.num_steps = num_steps
        self.sigma_adjust = sigma_adjust  # Also should be at least 5 x num_dim
        self.covariance_adjust = covariance_adjust
        self.save_interval = save_interval
        self.accept_ratio = accept_ratio
        self._do_save = temp_dir is not None and save_interval is not None

        self.position_file = None
        self.burn_file = None
        self.chain_file = None
        self.covariance_file = None
        self.covariance_plot = None

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

        log_posterior = model.get_posterior
        start = model.get_start

        assert log_posterior is not None
        assert start is not None

        if uid is None:
            uid = "mh"
        self._update_temp_files(uid)
        self.save_dims = save_dims
        self.log_posterior = log_posterior
        self.start = start
        position, burnin, chain, covariance = self._load()
        if burnin is not None:
            self.logger.debug("Found burnin of size %d" % burnin.shape[0])
        if chain is not None:
            self.logger.debug("Found chain of size %d" % chain.shape[0])
        position = self._ensure_position(position)
        if self.save_dims is None:
            self.save_dims = len(position) - self.space

        if chain is None or burnin is None or burnin.shape[0] < self.num_burn:
            position, covariance, burnin = self._do_burnin(position, burnin, covariance)
            chain = None

        if self.do_plot_covariance:
            self.plot_covariance(burnin)

        c, w, p = self._do_chain(position, covariance, chain=chain)
        self.logger.info("Returning results")
        return {"chain": c, "weights": w, "posterior": p}

    def _do_burnin(self, position, burnin, covariance):
        if burnin is None:
            # Initialise burning to all zeros. 2 from posterior and step size
            burnin = np.zeros((self.num_burn, position.size))
            step = 1
            burnin[0, :] = position
        elif burnin.shape[0] < self.num_burn:
            step = burnin.shape[0]
            position = burnin[-1, :]
            # If we only saved part of the burnin to save size, add the rest in as zeros
            burnin = np.vstack((burnin, np.zeros((self.num_burn - burnin.shape[0], position.size))))
        else:
            step = self.num_burn
        num_dim = position.size - self.space
        if covariance is None:
            covariance = np.identity(position.size - self.space)

        last_save_time = time()
        self.logger.info("Starting burn in")
        while step < self.num_burn:
            # If sigma adjust, adjust
            if step % self.sigma_adjust == 0 and step > 0:
                burnin[step - 1, self.IND_S] = self._adjust_sigma_ratio(burnin, step)
            # If covariance adjust, adjust
            if step % self.covariance_adjust == 0 and step > 0 and step > num_dim * 10:
                covariance = self._adjust_covariance(burnin, step)
                burnin[step - 1, self.IND_S] = 1.0

            # Get next step
            burnin[step, :], weight = self._get_next_step(burnin[step - 1, :], covariance, burnin=True)
            burnin[step - 1, self.IND_W] = weight
            if self.callback is not None:
                self.callback(
                    burnin[step - 1, self.IND_P],
                    burnin[step - 1, self.space : self.space + self.save_dims],
                    weight=burnin[step - 1, self.IND_W],
                )
            step += 1
            if step == self.num_burn or (self._do_save and (time() - last_save_time) > self.save_interval):
                self._save(burnin[step - 1, :], burnin[:step, :], None, covariance)
                last_save_time = time()

        return burnin[-1, :], covariance, burnin

    def _do_chain(self, position, covariance, chain=None):
        dims = self.save_dims if self.save_dims is not None else position.size - self.space
        size = dims + self.space
        if chain is None:
            current_step = 1
            chain = np.zeros((self.num_steps, size))
            chain[0, :] = position[:size]
        elif chain.shape[0] < self.num_steps:
            current_step = chain.shape[0]
            chain = np.vstack((chain, np.zeros((self.num_steps - chain.shape[0], size))))
        else:
            current_step = self.num_steps

        last_save_time = time()
        self.logger.info("Starting chain")
        while current_step < self.num_steps:
            position, weight = self._get_next_step(position, covariance)
            chain[current_step, :] = position[:size]
            chain[current_step - 1, self.IND_W] = weight
            if self.callback is not None:
                self.callback(
                    chain[current_step - 1, self.IND_P], chain[current_step - 1, self.space :], weight=chain[current_step - 1, self.IND_W]
                )
            current_step += 1
            if current_step == self.num_steps or (self._do_save and (time() - last_save_time) > self.save_interval):
                self._save(position, None, chain[:current_step, :], None)
                last_save_time = time()
        return chain[:, self.space :], chain[:, self.IND_W], chain[:, self.IND_P]

    def _update_temp_files(self, uid):
        if self.temp_dir is not None:
            self.position_file = self.temp_dir + os.sep + "%s_mh_position.npy" % uid
            self.burn_file = self.temp_dir + os.sep + "%s_mh_burn.npy" % uid
            self.chain_file = self.temp_dir + os.sep + "%s_mh_chain.npy" % uid
            self.covariance_file = self.temp_dir + os.sep + "%s_mh_covariance.npy" % uid
            self.covariance_plot = self.temp_dir + os.sep + "%s_mh_covariance.png" % uid

    def _ensure_position(self, position):
        """Ensures that the position object, which can be none from loading, is a
        valid [starting] position.
        """

        if position is None:
            final_pos = None
            if not callable(self.start):
                final_pos = self.start
                v = self.log_posterior(*final_pos)
            else:
                final_pos = list(self.start(num_walkers=1).flatten())
                v = self.log_posterior(final_pos)
            position = np.concatenate(([v, 0.1, 1], final_pos))
        return position

    def _adjust_sigma_ratio(self, burnin, index):
        subsection = burnin[index - self.sigma_adjust : index, :]
        actual_ratio = 1 / np.average(subsection[:, self.IND_W])

        sigma_ratio = burnin[index - 1, self.IND_S]
        ratio = self.accept_ratio / actual_ratio
        dims = burnin.shape[1] - self.space
        adjust_amount = np.power(ratio, 1.0 / dims)
        sigma_ratio /= adjust_amount
        # if actual_ratio < self.accept_ratio:
        #     sigma_ratio *= 0.9
        # else:
        #     sigma_ratio /= 0.9
        self.logger.debug(
            "Adjusting sigma: Want %0.3f, got %0.3f. " "Updating ratio to %0.5f" % (self.accept_ratio, actual_ratio, sigma_ratio)
        )
        return sigma_ratio

    def _adjust_covariance(self, burnin, index, return_cov=False):
        params = burnin.shape[1] - self.space
        if params == 1:
            return np.ones((1, 1))
        subset = burnin[int(np.floor(index / 2)) : index, :]
        covariance = np.cov(subset[:, self.space :].T, fweights=subset[:, self.IND_W])

        # import matplotlib.pyplot as plt
        # plt.imshow(covariance, cmap="viridis")
        # plt.show()
        if return_cov:
            return covariance
        res = np.linalg.cholesky(covariance)
        self.logger.debug("Adjusting covariance and resetting sigma ratio")
        return res

    def _propose_point(self, position, covariance):
        p = position[self.space :]
        eta = np.random.normal(size=p.size)
        step = np.dot(covariance, eta) * position[self.IND_S]
        return p + step

    def _get_next_step(self, position, covariance, burnin=False):
        attempts = 1
        counter = 1
        past_pot = position[self.IND_P]
        while True:
            pot = self._propose_point(position, covariance)
            posterior = self.log_posterior(pot)
            if posterior > past_pot or np.exp(posterior - past_pot) > np.random.uniform():
                result = np.concatenate(([posterior, position[self.IND_S], 1], pot))
                return result, attempts
            else:
                attempts += 1
                counter += 1
                if counter > 100 and burnin:
                    position[self.IND_S] *= 0.9
                    counter = 0

    def load_file(self, filename):
        result = np.load(filename)
        posterior = result[:, MetropolisHastings.IND_P]
        weight = result[:, MetropolisHastings.IND_W]
        chain = result[:, MetropolisHastings.space :]
        return {"posterior": posterior, "weights": weight, "chain": chain}

    def _load(self):
        position = None
        if self.position_file is not None and os.path.exists(self.position_file):
            position = np.load(self.position_file)
        burnin = None
        if self.burn_file is not None and os.path.exists(self.burn_file):
            burnin = np.load(self.burn_file)
        chain = None
        if self.chain_file is not None and os.path.exists(self.chain_file):
            chain = np.load(self.chain_file)
        covariance = None
        if self.covariance_file is not None and os.path.exists(self.covariance_file):
            covariance = np.load(self.covariance_file)
        return position, burnin, chain, covariance

    def _save(self, position, burnin, chain, covariance):
        if position is not None and self.position_file is not None:
            np.save(self.position_file, position)
        if burnin is not None and self.burn_file is not None:
            self.logger.info("Serialising results to file. Burnin has %d steps" % burnin.shape[0])
            np.save(self.burn_file, burnin.astype(np.float32))
        if chain is not None and self.chain_file is not None:
            self.logger.info("Serialising results to file. Chain has %d steps" % chain.shape[0])
            np.save(self.chain_file, chain.astype(np.float32))
        if covariance is not None and self.covariance_file is not None:
            np.save(self.covariance_file, covariance)

    def plot_covariance(self, burnin):
        if self.covariance_plot is None:
            return
        import matplotlib.pyplot as plt

        covariance = self._adjust_covariance(burnin, burnin.shape[0], return_cov=True)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        h = ax[0].imshow(covariance, cmap="viridis", interpolation="none")
        # div1 = make_axes_locatable(ax[0])
        # cax1 = div1.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(h, cax=cax1)

        diag = np.diag(1 / np.sqrt(np.diag(covariance)))
        cor = np.dot(np.dot(diag, covariance), diag)
        h2 = ax[1].imshow(cor, cmap="viridis", interpolation="none")
        # div2 = make_axes_locatable(ax[1])
        # cax2 = div2.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(h2, cax=cax2)
        fig.gca().set_frame_on(False)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        self.logger.info("Saving covariance plot")
        fig.savefig(self.covariance_plot, bbox_inches="tight", dpi=300)
        plt.close(fig)
        import gc

        gc.collect(2)
