import logging
import sys
sys.path.append("../..")
from barry.config.base import setup
from barry.framework.fitter import Fitter
from barry.framework.datasets.mock_correlation import MockAverageCorrelations
from barry.framework.models.bao_correlation_poly import CorrelationPolynomial

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    pfn, dir_name, file = setup(__file__)

    model = CorrelationPolynomial()
    data = MockAverageCorrelations()

    fitter = Fitter(dir_name, max_steps=20000, burnin=10000)
    fitter.set_models(model)
    fitter.set_data(data)
    fitter.set_num_walkers(30)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer
        res,  = fitter.load()

        posterior, weight, chain = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        # c.plotter.plot(filename=pfn + "_contour.png")

        split_walkers = True
        if split_walkers:
            res2 = fitter.load(split_walkers=True)
            c = ChainConsumer()
            for i, (posterior, weight, chain) in enumerate(res2):
                c.add_chain(chain, weights=weight, parameters=labels, name=f"Walker {i}")
            c.plotter.plot(filename=pfn + "_walkers.png")


