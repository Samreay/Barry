from time import time
import logging
import numpy as np
import os


class EmceeWrapper(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.logger = logging.getLogger("barry")
        self.chain = None
        self.posterior = None

    def run_chain(self, num_steps, num_burn, num_walkers, num_dim, start=None, save_interval=300,
                  save_dim=None, temp_dir=None, uid="ensemble"):
        assert num_steps > num_burn, "num_steps has to be larger than num_burn"
        if save_dim is not None:
            assert save_dim <= num_dim, "You cannot save more dimensions than you actually have"
        else:
            save_dim = num_dim

        past_chain = None
        pos = None
        if temp_dir is not None:
            self.logger.debug("Looking in temp dir %s" % temp_dir)
            chain_file = os.path.join(temp_dir, uid + "_ens.chain.npy")
            position_file = os.path.join(temp_dir, uid + "_ens.pos.npy")
            posterior_file = os.path.join(temp_dir, uid + "_ens.prob.npy")
            try:
                pos = np.load(position_file)
                past_chain = np.load(chain_file)
                past_posterior = np.load(posterior_file)
                self.logger.info("Found chain of %d steps" % past_chain.shape[1])
            except IOError:
                self.logger.info("Prior chain and/or does not exist. Looked in %s" % position_file)

        if start is None and pos is None:
            raise ValueError("You need to have either a starting function or existing chains")

        if pos is None:
            pos = np.array([start() for i in range(num_walkers)])

        step = 0
        self.chain = np.zeros((num_walkers, num_steps, save_dim))
        self.posterior = np.zeros((num_walkers, num_steps))
        if past_chain is not None and past_posterior is not None:
            step = min(past_chain.shape[1], num_steps)
            num = num_steps - step
            self.chain[:, :step, :] = past_chain[:, :step, :]
            self.posterior[:, :step] = past_posterior[:, :step]
            self.logger.debug("A further %d steps are required" % num)
        else:
            num = num_steps
            self.logger.debug("Running full chain of %d steps" % num)
        t = time()

        if step == num_steps:
            self.logger.debug("Returning serialised data from %s" % temp_dir)
            return self.get_results(num_burn)
        else:
            self.logger.debug("Starting sampling. Saving to %s ever %d seconds"
                              % (temp_dir, save_interval))

        for result in self.sampler.sample(pos, iterations=num, storechain=False):
            self.chain[:, step, :] = result[0][:, :save_dim]
            self.posterior[:, step] = result[1]
            step += 1
            if step == 1 or temp_dir is not None and save_interval is not None:
                t2 = time()
                if temp_dir is not None and \
                        (step == 1 or t2 - t > save_interval or step == num_steps):
                    t = t2
                    position = result[0]
                    np.save(position_file, position)
                    np.save(chain_file, self.chain[:, :step, :])
                    np.save(posterior_file, self.posterior[:, :step])
                    self.logger.debug("Saving chain with %d steps" % step)
        return self.get_results(num_burn)

    def get_results(self, num_burn):
        return self.chain[:, num_burn:, :].reshape((-1, self.chain.shape[2]))
