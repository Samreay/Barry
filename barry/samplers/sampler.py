import abc


class Sampler(object):
    __metaclass__ = abc.ABCMeta

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

    def load_file(self, filename):
        """Load existing results from a file"""
        raise NotImplementedError()
