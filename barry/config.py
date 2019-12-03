import logging
import os
import random
import shutil
import time
from functools import lru_cache
import inspect
import yaml


@lru_cache(maxsize=1)
def get_config():
    config_path = os.path.join(os.path.dirname(inspect.stack()[0][1]), "..", "config.yml")
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


def is_local():
    return shutil.which(get_config()["hpc_determining_command"]) is None


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)23s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
