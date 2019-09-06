import logging

from barry.cosmology.camb_generator import getCambGenerator
from barry.models import PowerNoda2019
from barry.postprocessing import BAOExtractor

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True

    c = getCambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s)

    model1 = PowerNoda2019(recon=recon, name=f"Noda2019, recon={recon}", postprocess=postprocess)

    from barry.datasets.mock_power import PowerSpectrum_SDSS_DR12_Z061_NGC
    from barry.datasets.dummy import DummyPowerSpectrum_SDSS_DR12_Z061_NGC
    dataset1 = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon, postprocess=postprocess, min_k=0.03, max_k=0.25)
    dataset2 = DummyPowerSpectrum_SDSS_DR12_Z061_NGC(name="Dummy data, real window fn", dummy_window=False, postprocess=postprocess)
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