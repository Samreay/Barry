import logging
import os
import shutil
import socket
import sys
import platform

import numpy as np

from barry.framework.doJob import write_jobscript_slurm
from barry.framework.samplers.metropolisHastings import MetropolisHastings
from barry.framework.samplers.viewer import Viewer


class Fitter(object):
    def __init__(self, temp_dir):
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.data = []
        self.num_realisations = 30
        self.num_walkers = 10
        self.num_cpu = None
        self.temp_dir = temp_dir
        self.max_steps = 10000
        self.set_num_cpu()
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

    def set_num_realisations(self, num_realisations):
        self.num_realisations = num_realisations
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            self.num_cpu = self.num_realisations * self.num_walkers
        else:
            self.num_cpu = num_cpu

    def set_num_walkers(self, num_walkers):
        self.num_walkers = num_walkers
        return self

    def get_num_jobs(self):
        num_jobs = len(self.models) * len(self.data) * self.num_realisations * self.num_walkers
        return num_jobs

    def get_indexes_from_index(self, index):
        num_simulations = len(self.data)
        num_cosmo = self.num_realisations
        num_walkers = self.num_walkers

        num_per_model_sim = num_cosmo * num_walkers
        num_per_model = num_simulations * num_per_model_sim

        model_index = index // num_per_model
        index -= model_index * num_per_model
        sim_index = index // num_per_model_sim
        index -= sim_index * num_per_model_sim
        cosmo_index = index // num_walkers
        walker_index = index % num_walkers

        return model_index, sim_index, cosmo_index, walker_index

    def run_fit(self, model_index, data_index, realisation_index, walker_index, full=True):
        model = self.models[model_index]
        data = self.data[data_index].get_data()

        model.set_data(data)

        uid = f"chain_{model_index}_{data_index}_{realisation_index}_{walker_index}"

        debug = not full
        if full:
            w, n = 3000, self.max_steps
        else:
            w, n = 3000, 7000

        callback = None
        if debug:
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

    def fit(self, file):

        num_jobs = self.get_num_jobs()
        num_models = len(self.models)
        num_simulations = len(self.data)
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (num_models, num_simulations, self.num_realisations, self.num_walkers, num_jobs))

        if self.is_laptop():
            self.logger.info("Running locally on the 0th index.")
            self.run_fit(0, 0, 0, 0, full=False)
        else:
            if len(sys.argv) == 1:
                h = socket.gethostname()
                partition = "regular" if "edison" in h else "smp"
                if os.path.exists(self.temp_dir):
                    self.logger.info("Deleting %s" % self.temp_dir)
                    shutil.rmtree(self.temp_dir)
                filename = write_jobscript_slurm(file, name=os.path.basename(file),
                                                 num_tasks=self.get_num_jobs(), num_cpu=self.num_cpu,
                                                 delete=True, partition=partition)
                self.logger.info("Running batch job at %s" % filename)
                os.system("sbatch %s" % filename)
            else:
                index = int(sys.argv[1])
                mi, si, ci, wi = self.get_indexes_from_index(index)
                self.logger.info("Running model %d, sim %d, cosmology %d, walker number %d" % (mi, si, ci, wi))
                self.run_fit(mi, si, ci, wi)

    def load_file(self, file):
        data = np.load(file)
        return data

    def load(self, split_models=True, split_sims=True, split_cosmo=False):
        files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith("_chain.npy")])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        sim_indexes = [int(f.split("_")[2]) for f in files]
        cosmo_indexes = [int(f.split("_")[3]) for f in files]
        chains = [self.load_file(f) for f in filenames]

        results = []
        prev_model, prev_sim, prev_cosmo = 0, 0, 0
        stacked = None
        for c, mi, si, ci in zip(chains, model_indexes, sim_indexes, cosmo_indexes):
            if (prev_cosmo != ci and split_cosmo) or (prev_model != mi and split_models) or (prev_sim != si and split_sims):
                if stacked is not None:
                    results.append(stacked)
                stacked = None
                prev_model = mi
                prev_sim = si
                prev_cosmo = ci
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