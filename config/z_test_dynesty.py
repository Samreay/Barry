from barry.config import setup
from barry.fitter import Fitter
from barry.models.test import TestModel
from barry.datasets.test import TestDataset
from barry.samplers import DynestySampler

if __name__ == "__main__":
    import sys

    sys.path.append("..")

    pfn, dir_name, file = setup(__file__)

    model = TestModel()
    data = TestDataset()

    sampler = DynestySampler(temp_dir=dir_name, max_iter=None)

    fitter = Fitter(dir_name)
    fitter.add_model_and_dataset(model, data)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)

    if fitter.should_plot():

        res, = fitter.load()

        posterior, weight, chain, evidence, model, data, extra = res
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(weight)
        ax[1].plot(evidence)
        plt.show()

        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=model.get_labels())
        c.plotter.plot(filename=pfn + "_contour.png")
