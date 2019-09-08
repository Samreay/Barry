import logging
import os
import random
import time
from functools import lru_cache
import inspect
import numpy as np
import yaml


@lru_cache(maxsize=1)
def get_config():
    config_path = os.path.join(os.path.dirname(inspect.stack()[0][1]), "config.yml")
    assert os.path.exists(config_path), f"File {config_path} cannot be found."
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup(filename):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s |%(funcName)18s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plot_dir = os.path.dirname(os.path.abspath(filename)) + "/plots/%s/" % os.path.basename(filename)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(filename)[:-3]
    file = os.path.abspath(filename)
    time.sleep(random.random())
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    except Exception:
        pass
    return pfn, dir_name, file


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)
