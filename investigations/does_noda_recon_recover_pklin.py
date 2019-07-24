import logging

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.models import PowerBeutler2017
from barry.framework.postprocessing import BAOExtractor

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True

    c = CambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s)

    model1 = PowerBeutler2017(recon=recon, name=f"Noda2019, recon={recon}", postprocess=postprocess)

    from barry.framework.datasets.mock_power import MockSDSSdr12PowerSpectrum
    from barry.framework.datasets.dummy_power import DummyPowerSpectrum
    dataset1 = MockSDSSdr12PowerSpectrum(name="SDSS Recon mean", recon=recon, min_k=0.02, max_k=0.3, reduce_cov_factor=31.62, step_size=5, postprocess=postprocess)
    dataset2 = DummyPowerSpectrum(name="Dummy data, real window fn", min_k=0.02, max_k=0.3, step_size=5, dummy_window=False, postprocess=postprocess)
    data1 = dataset1.get_data()
    data2 = dataset2.get_data()

    # First comparison - the actual recon data
    model1.set_data(data1)
    p, minv = model1.optimize()
    print(p)
    print(minv)
    model1.plot(p)

    # The second comparison, dummy data with real window function
    model1.set_data(data2)
    p, minv = model1.optimize()
    print(p)
    print(minv)
    model1.plot(p)

    # FINDINGS
    # 0.993 for the SDSS data, 0.978 for Dummy data. Bit odd.