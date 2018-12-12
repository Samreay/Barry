import abc


class GenericSampler(object):
    __metaclass__ = abc.ABCMeta

    def fit(self, log_posterior, start, save_dims=None, uid=None):
        """" Runs the sampler over the model and returns the flat chain of results

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
            A dictionary of results containing:
                - *chain*: the chain
                - *weights*: chain weights if applicable
        """
        raise NotImplementedError()

    def load_file(self, filename):
        """ Load existing results from a file"""
        raise NotImplementedError()