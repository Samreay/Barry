import logging
import os
import shutil
import socket
import sys
import numpy as np

from barry.config import get_config
from barry.doJob import write_jobscript_slurm
from barry.samplers import DynestySampler, GridSearch
from barry.utils import get_hpc


class Fitter(object):
    """This class manages all the model fitting you'll be doing.

    You simply declare pairs of models and datasets that you want to fit,
    optionally tell it which sampler to use and then tell it do its job.

    Hopefully minimal fuss involved.

    """

    def __init__(self, temp_dir, save_dims=None, remove_output=True):
        """

        Parameters
        ----------
        temp_dir : str
            The directory into which plots and output chains will be stored
        save_dims : int, optional
            To reduce filesize if needed, you can an arbitrary amount
            of dimensions when putting chains to disk. So a two here would
            save the first two chains and not the rest. By default, saves
            everything.
        remove_output : bool, optional
            Whether or not to remove old output when running this on a cluster
            computer. By default, this is true, such that you can rerun files
            without worrying about getting files mixed up. But if, for example,
            you hit a walltime limit and one of your thousand jobs gets killed,
            set this to false and rerun, and it will only run the failed job.
        """
        self.logger = logging.getLogger("barry")
        self.model_datasets = []
        self.num_walkers = 10
        self.num_concurrent = None
        self.temp_dir = temp_dir
        self.sampler = None
        self.save_dims = save_dims
        self.remove_output = remove_output
        os.makedirs(temp_dir, exist_ok=True)
        if not remove_output:
            self.logger.warning("OUTPUT IS NOT BEING REMOVED, BE WARNED IF THIS IS SUPPOSED TO BE A FRESH RUN")

    def add_model_and_dataset(self, model, dataset, **extra_args):
        """Adds a model-dataset pair to fit.

        Parameters
        ----------
        model : `barry.models.Model`
            The model class to fit.
        dataset : `barry.datasets.Dataset`
            The dataset to fit.
        extra_args : kwargs, optional
            Any extra information you want returned with the chains from model fitting.
            I often use this to name my pairs to make it convenient to load into `ChainConsumer`.

        """

        if "realisation" not in extra_args:
            extra_args["realisation"] = dataset.realisation
        if "name" not in extra_args:
            extra_args["name"] = dataset.get_name() + " + " + model.get_name()
        self.model_datasets.append((model, dataset.get_data(), extra_args))

    def set_num_concurrent(self, num_concurrent=None):
        """Set the number of jobs allowed to run in the job array at once.

        Parameters
        ----------
        num_concurrent : int, optional
            The maximum number of concurrent jobs to have. Defaults to *all of them*.
            Not changing this and submitting a huge job array may make your sysadmin angry.
        """
        self.num_concurrent = num_concurrent

    def get_num_concurrent(self):
        """Gets the number of current jobs limit.

        Returns
        -------
        num_concurrent : int
        """
        if self.num_concurrent is None:
            return len(self.model_datasets) * self.num_walkers
        return self.num_concurrent

    def set_num_walkers(self, num_walkers):
        """Sets the number of walks for each model-dataset pair.

        Ie, how many different runs we should do for each pair to ensure convergence
        and good statistics. Setting this to 10 for the MH sampling for example would
        say to start ten independent walks and combine them all at the end.

        Parameters
        ----------
        num_walkers : int
        """
        self.num_walkers = num_walkers

        return self

    def get_num_jobs(self):
        """Gets the total number of jobs that wil be submitted.

        Returns
        -------
        num_jobs : int
        """
        num_jobs = len(self.model_datasets) * self.num_walkers
        return num_jobs

    def _get_indexes_from_index(self, index):
        model_index = index // self.num_walkers
        walker_index = index % self.num_walkers
        return model_index, walker_index

    def set_sampler(self, sampler):
        """Sets the sampler

        Parameters
        ----------
        sampler : `barry.samplers.sampler.Sampler`

        """
        self.sampler = sampler

        return self

    def get_sampler(self):
        """Returns the sampler. If not set, creates a DynestySampler

        Returns
        -------
        sampler : `barry.samplers.sampler.Sampler`
        """
        if self.sampler is None:
            self.sampler = DynestySampler(temp_dir=self.temp_dir, nlive=500)
        return self.sampler

    def _run_fit(self, model_index, walker_index):

        model = self.model_datasets[model_index][0]
        data = self.model_datasets[model_index][1]

        model.set_data(data)
        uid = f"chain_{model_index}_{walker_index}"

        sampler = self.get_sampler()

        self.logger.info("Running fitting job, saving to %s" % self.temp_dir)
        self.logger.info(f"\tModel is {model}")
        self.logger.info(f"\tData is {' '.join([d['name'] for d in self.model_datasets[model_index][1]])}")
        sampler.fit(model, uid=uid, save_dims=self.save_dims)
        self.logger.info("Finished sampling")

    def is_local(self):
        return shutil.which(get_config()["hpc_determining_command"]) is None

    def is_interactive(self):
        import __main__ as main

        return not hasattr(main, "__file__")

    def should_plot(self):
        # Plot if we're running on the laptop, or we've passed "plot" as the argument
        # to the python script on the HPC
        return self.is_local() or (len(sys.argv) == 2 and sys.argv[1] == "plot")

    def fit(self, file, index=0):
        num_concurrent = self.get_num_concurrent()

        num_jobs = self.get_num_jobs()
        num_models = len(self.model_datasets)
        self.logger.info(f"With {num_models} models+datasets and {self.num_walkers} walkers, " f"have {num_jobs} jobs")

        if self.is_local() or self.is_interactive():
            mi, wi = self._get_indexes_from_index(index)
            self.logger.info("Running model_dataset %d, walker number %d" % (mi, wi))
            self._run_fit(mi, wi)
        else:
            if len(sys.argv) == 1:
                # if launching the job for the first time
                if os.path.exists(self.temp_dir):
                    if self.remove_output:
                        self.logger.info("Deleting %s" % self.temp_dir)
                        shutil.rmtree(self.temp_dir)
                hpc = get_hpc()
                filename = write_jobscript_slurm(
                    file, name=os.path.basename(file), num_tasks=self.get_num_jobs(), num_concurrent=num_concurrent, delete=False, hpc=hpc
                )
                self.logger.info("Running batch job at %s" % filename)
                config = get_config()
                os.system(f"{config['hpc_submit_command']} {filename}")
            else:
                # or if running a specific fit to a model+dataset pair
                if sys.argv[1].isdigit():
                    index = int(sys.argv[1])
                else:
                    index = -1
                if index != -1 and index < self.get_num_jobs():
                    mi, wi = self._get_indexes_from_index(index)
                    self.logger.info("Running model_dataset %d, walker number %d" % (mi, wi))
                    self._run_fit(mi, wi)

    def _load_file(self, file):
        d = self.get_sampler().load_file(file)
        chain = d["chain"]
        weights = d.get("weights")
        evidence = d.get("evidence")
        chi2 = d.get("chi2")
        if weights is None:
            weights = np.ones((chain.shape[0], 1))
        if evidence is None:
            evidence = np.full(chain.shape[0], np.nan)
        if chi2 is None:
            chi2 = np.full(chain.shape[0], np.nan)
        if len(weights.shape) == 1:
            weights = np.atleast_2d(weights).T
        if len(evidence.shape) == 1:
            evidence = np.atleast_2d(evidence).T
        if len(chi2.shape) == 1:
            chi2 = np.atleast_2d(chi2).T
        posterior = d.get("posterior")
        if posterior is None:
            posterior = np.ones((chain.shape[0], 1))
        if len(posterior.shape) == 1:
            posterior = np.atleast_2d(posterior).T
        result = np.hstack((posterior, weights, evidence, chi2, chain))
        return result

    def load(self, split_models=True, split_walkers=False):
        """Load in all the chains and fitting results

        Parameters
        ----------
        split_models : bool, optional
            Keep the models split and separate. Set this to false to combine the chains (very specific user
            case for this, think hard if you feel like you want to set it to `False`).
        split_walkers : bool, optional
            Split up each walker to make things like convergence diagnostics easier. Defaults to `False`

        Returns
        -------
            fits : list
                A list of each model+dataset pair (assuming `split_models` is `True`).
                Each element contains, in order:
                    - log_posterior[steps]
                    - weights[steps] (not log)
                    - chain[steps:dimensions]
                    - the model
                    - the dataset
                    - dict containing any `extra` information passed in.
        """
        self.logger.info("Loading chains")
        files = [f for f in os.listdir(self.temp_dir) if f.endswith("chain.npy")]
        files.sort(key=lambda s: [int(s.split("_")[1]), int(s.split("_")[2])])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        walker_indexes = [int(f.split("_")[2]) for f in files]
        chains = [self._load_file(f) for f in filenames]

        results = []
        results_models = []
        prev_model, prev_walkers = 0, 0
        stacked = None

        to_sort = np.array(model_indexes) + 0.0001 * np.array(walker_indexes)
        sorted_indexes = np.argsort(to_sort)

        for index in sorted_indexes:
            c, mi, wi = chains[index], model_indexes[index], walker_indexes[index]
            if (prev_walkers != wi and split_walkers) or (prev_model != mi and split_models):
                if stacked is not None:
                    results.append(stacked)
                    results_models.append(self.model_datasets[prev_model])
                stacked = None
                prev_model = mi
                prev_walkers = wi
            if stacked is None:
                stacked = c
            else:
                stacked = np.vstack((stacked, c))
        results_models.append(self.model_datasets[mi])
        results.append(stacked)

        finals = []
        for result, model in zip(results, results_models):
            posterior = result[:, 0]
            weight = result[:, 1]
            evidence = result[:, 2]
            chi2 = result[:, 3]
            chain = result[:, 4:]
            if np.any(~np.isnan(chi2)):
                finals.append((posterior, weight, chain, evidence, chi2, model[0], model[1], model[2]))
            else:
                finals.append((posterior, weight, chain, evidence, model[0], model[1], model[2]))
        self.logger.info(f"Loaded {len(finals)} chains")
        if len(finals) == 1:
            self.logger.info(f"Chain has shape {finals[0][2].shape}")
        return finals
