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
    def __init__(self, temp_dir):
        self.logger = logging.getLogger("barry")
        self.models = []
        self.data = []
        self.num_walkers = 10
        self.num_cpu = None
        self.temp_dir = temp_dir
        self.sampler = None
        os.makedirs(temp_dir, exist_ok=True)

    def set_models(self, *models):
        self.models = models
        return self

    def set_data(self, *data):
        self.data = data
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            num_cpu = len(self.models) * len(self.data) * self.num_walkers
        self.num_cpu = num_cpu

    def get_num_cpu(self):
        if self.num_cpu is None:
            self.set_num_cpu()
        return self.num_cpu

    def set_num_walkers(self, num_walkers):
        self.num_walkers = num_walkers
        return self

    def get_num_jobs(self):
        num_jobs = len(self.models) * len(self.data) * self.num_walkers
        return num_jobs

    def get_indexes_from_index(self, index):
        num_simulations = len(self.data)
        num_walkers = self.num_walkers

        num_per_model = num_simulations * num_walkers

        model_index = index // num_per_model
        index -= model_index * num_per_model
        sim_index = index // num_walkers
        index -= sim_index * num_walkers
        walker_index = index % num_walkers

        return model_index, sim_index, walker_index

    def set_sampler(self, sampler):
        self.sampler = sampler

    def get_sampler(self, full=True, show_viewer=False, model_index=None):
        if self.sampler is None:
            callback = None
            if show_viewer:
                from barry.framework.samplers.viewer import Viewer
                model = self.models[model_index]
                viewer = Viewer(model.get_extents(), parameters=model.get_labels())
                callback = viewer.callback

            debug = not full
            self.sampler = MetropolisHastings(num_burn=5000, num_steps=10000, temp_dir=self.temp_dir,
                                              callback=callback, plot_covariance=debug)
        return self.sampler

    def run_fit(self, model_index, data_index, walker_index, full=True, show_viewer=False):
        model = self.models[model_index]
        data = self.data[data_index].get_data()

        model.set_data(data)

        uid = f"chain_{model_index}_{data_index}_{walker_index}"

        sampler = self.get_sampler(full=full, show_viewer=show_viewer, model_index=model_index)

        self.logger.info("Running fitting job, saving to %s" % self.temp_dir)

        sampler.fit(model.get_posterior, model.get_start, uid=uid)
        # Perform the fitting here
        # Save results out

        self.logger.info("Finished sampling")

    def is_laptop(self):
        return "centos" not in platform.platform()

    def fit(self, file, viewer=False):
        if self.num_cpu is None:
            self.set_num_cpu()

        num_jobs = self.get_num_jobs()
        num_models = len(self.models)
        num_simulations = len(self.data)
        self.logger.info(f"With {num_models} models, {num_simulations} simulations and {self.num_walkers} walkers, "
                         f"have {num_jobs} jobs")

        if self.is_laptop():
            self.logger.info("Running locally on the 0th index.")
            self.run_fit(0, 0, 0, full=False, show_viewer=viewer)
        else:
            if len(sys.argv) == 1:
                h = socket.gethostname()
                partition = "regular" if "edison" in h else "smp"
                if os.path.exists(self.temp_dir):
                    self.logger.info("Deleting %s" % self.temp_dir)
                    shutil.rmtree(self.temp_dir)
                filename = write_jobscript_slurm(file, name=os.path.basename(file),
                                                 num_tasks=self.get_num_jobs(), num_cpu=self.get_num_cpu(),
                                                 delete=True, partition=partition)
                self.logger.info("Running batch job at %s" % filename)
                os.system("sbatch %s" % filename)
            else:
                index = int(sys.argv[1])
                mi, si, wi = self.get_indexes_from_index(index)
                self.logger.info("Running model %d, sim %d, walker number %d" % (mi, si, wi))
                self.run_fit(mi, si, wi)

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

    def load(self, split_models=True, split_sims=True, split_walkers=False):
        self.logger.info("Loading chains")
        files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith("chain.npy")])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        sim_indexes = [int(f.split("_")[2]) for f in files]
        walker_indexes = [int(f.split("_")[3]) for f in files]
        chains = [self.load_file(f) for f in filenames]

        results = []
        results_models, results_data = [], []
        prev_model, prev_sim, prev_walkers = 0, 0, 0
        stacked = None
        for c, mi, si, wi in zip(chains, model_indexes, sim_indexes, walker_indexes):
            if (prev_walkers != wi and split_walkers) or (prev_model != mi and split_models) or (prev_sim != si and split_sims):
                if stacked is not None:
                    results.append(stacked)
                    results_models.append(self.models[prev_model])
                    results_data.append(self.data[prev_sim])
                stacked = None
                prev_model = mi
                prev_sim = si
                prev_walkers = wi
            if stacked is None:
                stacked = c
            else:
                stacked = np.vstack((stacked, c))
        results_models.append(self.models[mi])
        results_data.append(self.data[si])
        results.append(stacked)

        finals = []
        for result, model, sim in zip(results, results_models, results_data):
            posterior = result[:, 0]
            weight = result[:, 1]
            chain = result[:, 2:]
            finals.append((posterior, weight, chain, model, sim))
        self.logger.info(f"Loaded {len(finals)} chains")
        if len(finals) == 1:
            self.logger.info(f"Chain has shape {finals[0][2].shape}")
        return finals