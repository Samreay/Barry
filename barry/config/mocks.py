from barry.config.base import setup
from barry.framework.fitter import Fitter
from barry.framework.models.bao import BAOModel
from barry.framework.simulations.mock import Mock

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    model = BAOModel()
    data = Mock()

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(data)
    fitter.set_num_realisations(1)
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
