from barry.config.base import setup
from barry.framework.fitter import Fitter
from barry.framework.models.test import TestModel
from barry.framework.datasets.test import TestDataset

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    model = TestModel()
    data = TestDataset()

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_data(data)
    fitter.set_num_walkers(2)
    fitter.fit(file)

    if fitter.is_laptop():  # As I'm not sure if the cluster has matplotlib
        from chainconsumer import ChainConsumer
        res, = fitter.load()

        posterior, weight, chain = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        c.plotter.plot(filename=pfn + "_contour.png")
