import logging
import os
import time
import random


def setup(filename):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s |%(funcName)18s]   %(message)s")
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
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
