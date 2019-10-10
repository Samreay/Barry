import logging
from barry.cosmology.camb_generator import getCambGenerator
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.models import PowerNoda2019
from barry.postprocessing import BAOExtractor

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    c = getCambGenerator()
    r_s = c.get_data()[0]

    postprocess = BAOExtractor(r_s, mink=0.15)

    for recon in [True, False]:

        model1 = PowerNoda2019(recon=recon, name=f"Noda2019, recon={recon}", postprocess=postprocess, fix_params=["om", "f", "gamma"])
        dataset1 = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon, postprocess=postprocess, min_k=0.03, max_k=0.15)
        data1 = dataset1.get_data()
        print(list(data1[0].keys()))
        print(data1[0]["ks_output"])
        exit()
        # First comparison - the actual recon data
        model1.set_data(data1)
        p, minv = model1.optimize()
        print(recon)
        print(p)
        print(minv)
        model1.plot(p)

        # FINDINGS
        # 2.022 for Recon, 2.092 for prerecon

        # Private communiucation with the team means for the mock means they found 1.86996 for recon and 1.89131 for prerecon
