import logging
import os
import shutil
import socket
import sys
import platform

import numpy as np

from barry.framework.doJob import write_jobscript_slurm
from barry.framework.samplers.metropolisHastings import MetropolisHastings


class Fitter(object):
    def __init__(self, temp_dir, save_dims=None, remove_output=True):
        self.logger = logging.getLogger("barry")
        self.model_datasets = []
        self.num_walkers = 10
        self.num_cpu = None
        self.temp_dir = temp_dir
        self.sampler = None
        self.save_dims = save_dims
        self.remove_output = remove_output
        os.makedirs(temp_dir, exist_ok=True)

    def add_model_and_dataset(self, model, dataset, **extra_args):
        self.model_datasets.append((model, dataset.get_data(), extra_args))
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            num_cpu = len(self.model_datasets) * self.num_walkers
        self.num_cpu = num_cpu

    def get_num_cpu(self):
        if self.num_cpu is None:
            self.set_num_cpu()
        return self.num_cpu

    def set_num_walkers(self, num_walkers):
        self.num_walkers = num_walkers
        return self

    def get_num_jobs(self):
        num_jobs = len(self.model_datasets) * self.num_walkers
        return num_jobs

    def get_indexes_from_index(self, index):
        model_index = index // self.num_walkers
        walker_index = index % self.num_walkers
        return model_index, walker_index

    def set_sampler(self, sampler):
        self.sampler = sampler

    def get_sampler(self, full=True):
        if self.sampler is None:
            debug = not full
            self.sampler = MetropolisHastings(num_burn=5000, num_steps=10000, temp_dir=self.temp_dir, plot_covariance=debug)
        return self.sampler

    def run_fit(self, model_index, walker_index, full=True):
        model = self.model_datasets[model_index][0]
        data = self.model_datasets[model_index][1]

        model.set_data(data)
        uid = f"chain_{model_index}_{walker_index}"

        sampler = self.get_sampler(full=full)

        self.logger.info("Running fitting job, saving to %s" % self.temp_dir)
        self.logger.info(f"Model is {model}")
        self.logger.info(f"Data is {self.model_datasets[model_index][1]['name']}")
        sampler.fit(model.get_posterior, model.get_start, uid=uid, save_dims=self.save_dims)
        # Perform the fitting here
        # Save results out

        self.logger.info("Finished sampling")

    def is_laptop(self):
        return "centos" not in platform.platform()

    def should_plot(self):
        return self.is_laptop() or (len(sys.argv) == 2 and int(sys.argv[1]) == -1)

    def fit(self, file):
        if self.num_cpu is None:
            self.set_num_cpu()

        num_jobs = self.get_num_jobs()
        num_models = len(self.model_datasets)
        self.logger.info(f"With {num_models} models+datasets and {self.num_walkers} walkers, "
                         f"have {num_jobs} jobs")

        if self.is_laptop():
            self.logger.info("Running locally on the 0th index.")
            self.run_fit(0, 0, full=False)
        else:
            if len(sys.argv) == 1:
                h = socket.gethostname()
                partition = "regular" if "edison" in h else "smp"
                if os.path.exists(self.temp_dir):
                    if self.remove_output:
                        self.logger.info("Deleting %s" % self.temp_dir)
                        shutil.rmtree(self.temp_dir)
                filename = write_jobscript_slurm(file, name=os.path.basename(file),
                                                 num_tasks=self.get_num_jobs(), num_cpu=self.get_num_cpu(),
                                                 delete=False, partition=partition)
                self.logger.info("Running batch job at %s" % filename)
                os.system("sbatch %s" % filename)
            else:
                index = int(sys.argv[1])
                if index != -1:
                    mi, wi = self.get_indexes_from_index(index)
                    self.logger.info("Running model_dataset %d, walker number %d" % (mi, wi))
                    self.run_fit(mi, wi)

    def load_file(self, file):
        d = self.get_sampler().load_file(file)
        chain = d["chain"]
        weights = d.get("weights")
        if weights is None:
            weights = np.ones((chain.shape[0], 1))
        if len(weights.shape) == 1:
            weights = np.atleast_2d(weights).T
        posterior = d.get("posterior")
        if posterior is None:
            posterior = np.ones((chain.shape[0], 1))
        if len(posterior.shape) == 1:
            posterior = np.atleast_2d(posterior).T
        result = np.hstack((posterior, weights, chain))
        return result

    def load(self, split_models=True, split_walkers=False):
        self.logger.info("Loading chains")
        files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith("chain.npy")])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        walker_indexes = [int(f.split("_")[2]) for f in files]
        chains = [self.load_file(f) for f in filenames]

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
            chain = result[:, 2:]
            finals.append((posterior, weight, chain, model[0], model[1], model[2]))
        self.logger.info(f"Loaded {len(finals)} chains")
        if len(finals) == 1:
            self.logger.info(f"Chain has shape {finals[0][2].shape}")
        return finals