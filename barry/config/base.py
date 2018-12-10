import logging
import os


def setup(filename):
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)22s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(filename)) + "/plots/%s/" % os.path.basename(filename)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(filename)[:-3]
    file = os.path.abspath(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return pfn, dir_name, file