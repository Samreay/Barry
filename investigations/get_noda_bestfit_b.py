import logging

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.framework.models import PowerNoda2019
from barry.framework.postprocessing import BAOExtractor, PureBAOExtractor

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    c = CambGenerator()
    r_s, _ = c.get_data()

    postprocess = BAOExtractor(r_s, mink=0.15)
    #postprocess = PureBAOExtractor(r_s)

    for recon in [True, False]:

        model1 = PowerNoda2019(recon=recon, name=f"Noda2019, recon={recon}", postprocess=postprocess, fix_params=["om", "gamma"])
        dataset1 = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon, postprocess=postprocess, min_k=0.03, max_k=0.15)
        data1 = dataset1.get_data()

        # First comparison - the actual recon data
        model1.set_data(data1)
        p, minv = model1.optimize()
        print(recon)
        print(p)
        print(minv)
        model1.plot(p)

        # FINDINGS
        # 1.988 for Recon, 2.01238 for prerecon