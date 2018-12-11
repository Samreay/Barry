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
    def __init__(self, temp_dir, max_steps=15000, burnin=5000):
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.data = []
        self.num_walkers = 10
        self.num_cpu = None
        self.temp_dir = temp_dir
        self.max_steps = max_steps
        self.burnin = burnin
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def set_models(self, *models):
        self.models = models
        return self

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def set_data(self, *data):
        self.data = data
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            self.num_cpu = len(self.models) * len(self.data) * self.num_walkers
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

    def run_fit(self, model_index, data_index, walker_index, full=True, show_viewer=False):
        model = self.models[model_index]
        data = self.data[data_index].get_data()

        model.set_data(data)

        uid = f"chain_{model_index}_{data_index}_{walker_index}"

        debug = not full
        if full:
            w, n = self.burnin, self.max_steps
        else:
            w, n = self.burnin, self.burnin

        callback = None
        if show_viewer:
            from barry.framework.samplers.viewer import Viewer
            viewer = Viewer(model.get_extents(), parameters=model.get_labels())
            callback = viewer.callback

        sampler = MetropolisHastings(num_burn=w, num_steps=n, temp_dir=self.temp_dir, callback=callback, plot_covariance=debug)

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
        data = np.load(file)
        return data

    def load(self, split_models=True, split_sims=True):
        files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith("_chain.npy")])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        sim_indexes = [int(f.split("_")[2]) for f in files]
        chains = [self.load_file(f) for f in filenames]

        results = []
        prev_model, prev_sim, prev_cosmo = 0, 0, 0
        stacked = None
        for c, mi, si in zip(chains, model_indexes, sim_indexes):
            if (prev_model != mi and split_models) or (prev_sim != si and split_sims):
                if stacked is not None:
                    results.append(stacked)
                stacked = None
                prev_model = mi
                prev_sim = si
            if stacked is None:
                stacked = c
            else:
                stacked = np.vstack((stacked, c))

        results.append(stacked)

        finals = []
        for result in results:
            posterior = result[:, MetropolisHastings.IND_P]
            weight = result[:, MetropolisHastings.IND_W]
            chain = result[:, MetropolisHastings.space:]
            finals.append((posterior, weight, chain))
        return finals