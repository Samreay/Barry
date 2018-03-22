import logging
import os
import shutil
import socket
import sys

from barry.framework.doJob import write_jobscript_slurm


class Fitter(object):
    def __init__(self, temp_dir):
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.simulations = []
        self.num_cosmologies = 30
        self.num_walkers = 10
        self.num_cpu = None
        self.temp_dir = temp_dir
        self.max_steps = 3000
        self.set_num_cpu()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def set_models(self, *models):
        self.models = models
        return self

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def set_simulations(self, *simulations):
        self.simulations = simulations
        return self

    def set_num_cosmologies(self, num_cosmologies):
        self.num_cosmologies = num_cosmologies
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            self.num_cpu = self.num_cosmologies * self.num_walkers
        else:
            self.num_cpu = num_cpu

    def set_num_walkers(self, num_walkers):
        self.num_walkers = num_walkers
        return self

    def get_num_jobs(self):
        num_jobs = len(self.models) * len(self.simulations) * self.num_cosmologies * self.num_walkers
        return num_jobs

    def get_indexes_from_index(self, index):
        num_simulations = len(self.simulations)
        num_cosmo = self.num_cosmologies
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

    def run_fit(self, model_index, simulation_index, cosmo_index, walker_index, full=True):
        model = self.models[model_index]
        data = self.simulations[simulation_index].get_data()

        out_file = self.temp_dir + "/chain_%d_%d_%d_%d.pkl" % (model_index, simulation_index, cosmo_index, walker_index)

        if full:
            w, n = 1000, self.max_steps
        else:
            w, n = 500, 1000

        self.logger.info("Running fitting job, saving to %s" % out_file)

        # Perform the fitting here
        # Save results out

        self.logger.info("Finished sampling")

    def is_laptop(self):
        return "science" in socket.gethostname()

    def fit(self, file):

        num_jobs = self.get_num_jobs()
        num_models = len(self.models)
        num_simulations = len(self.simulations)
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (num_models, num_simulations, self.num_cosmologies, self.num_walkers, num_jobs))

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