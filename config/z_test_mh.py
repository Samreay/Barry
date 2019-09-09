import sys

sys.path.append("..")
from barry.config import setup
from barry.fitter import Fitter
from barry.models.test import TestModel
from barry.datasets.test import TestDataset

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    model = TestModel()
    data = TestDataset()

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, data)
    fitter.set_num_walkers(2)
    fitter.fit(file)

    if fitter.should_plot():  # As I'm not sure if the cluster has matplotlib
        from chainconsumer import ChainConsumer

        res, = fitter.load()

        posterior, weight, chain, model, data = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        c.plotter.plot(filename=pfn + "_contour.png")
