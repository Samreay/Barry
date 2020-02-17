import sys

sys.path.append("..")
from barry.config import setup
from barry.fitter import Fitter
from barry.models.test import TestModel
from barry.datasets.test import TestDataset
from barry.samplers import MetropolisHastings

if __name__ == "__main__":
    pfn, dir_name, file = setup(__file__)

    model = TestModel()
    data = TestDataset()
    sampler = MetropolisHastings(num_steps=1000, num_burn=300, temp_dir=dir_name)

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, data)
    fitter.set_num_walkers(2)
    fitter.set_sampler(sampler)
    fitter.fit(file)

    if fitter.should_plot():
        from chainconsumer import ChainConsumer

        res, = fitter.load()

        posterior, weight, chain, evidence, model, data, extra = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        c.plotter.plot(filename=pfn + "_contour.png")
