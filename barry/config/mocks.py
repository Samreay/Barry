import logging
import os

from barry.framework.fitter import Fitter
from barry.framework.models.bao import BAOModel
from barry.framework.simulations.mock import Mock

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)22s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]
    file = os.path.abspath(__file__)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model = BAOModel()
    data = Mock()
    fitter = Fitter(dir_name)

    fitter.set_models(model)
    fitter.set_simulations(data)
    fitter.set_num_cosmologies(1)
    fitter.set_num_walkers(2)
    fitter.fit(file)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer
        res, = fitter.load()

        posterior, weight, chain = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        c.plotter.plot(filename=pfn + "_contour.png")
