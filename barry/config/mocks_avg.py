import logging
import sys
sys.path.append("../..")
import numpy as np
from barry.framework.samplers.ensemble import EnsembleSampler
from barry.config.base import setup
from barry.framework.fitter import Fitter
from barry.framework.datasets.mock_correlation import MockAverageCorrelations
from barry.framework.models.bao_correlation_poly import CorrelationPolynomial

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    pfn, dir_name, file = setup(__file__)

    model = CorrelationPolynomial()
    data = MockAverageCorrelations()
    sampler = EnsembleSampler(num_steps=1000, num_burn=500, temp_dir=dir_name, save_interval=300)

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_data(data)
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(10)
    fitter.fit(file, viewer=False)

    if fitter.is_laptop():
        from chainconsumer import ChainConsumer
        res,  = fitter.load()

        posterior, weight, chain = res
        labels = model.get_labels()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, parameters=labels)
        c.plotter.plot(filename=pfn + "_contour.png")

        if True:  # Plot each walker to check consistency
            res2 = fitter.load(split_walkers=True)
            c = ChainConsumer()
            for i, (posterior, weight, chain) in enumerate(res2):
                c.add_chain(chain, weights=weight, parameters=labels, name=f"Walker {i}")
            c.plotter.plot(filename=pfn + "_walkers.png")


