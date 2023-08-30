import os
import abc
import logging


class Sampler(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, temp_dir=None):

        self.logger = logging.getLogger("barry")
        self.num_steps = 1
        self.num_walkers = 1
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.autoconverge = False
        self.print_progress = False

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
        """
        raise NotImplementedError()

    def get_filename(self, uid):
        return os.path.join(self.temp_dir, f"{uid}_{self.get_file_suffix()}")

    def get_file_suffix(self):
        return "chain.npy"

    def load_file(self, filename):
        """Load existing results from a file"""
        raise NotImplementedError()
