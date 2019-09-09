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

    if fitter.should_plot():  # As I'm not sure if the cluster has matplotlib
        from chainconsumer import ChainConsumer

        res, = fitter.load()

        posterior, weight, chain, model, data, extra = res
        print(chain.shape, weight.shape)
        print(weight.max())
        import matplotlib.pyplot as plt

        plt.plot(weight)
        plt.show()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=model.get_labels())
        c.plotter.plot(filename=pfn + "_contour.png")
