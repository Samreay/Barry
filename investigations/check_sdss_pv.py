import numpy as np
import logging
from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_PV
from barry.models import PowerBeutler2017
from barry.models.model import Correction

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    dataset_data = PowerSpectrum_SDSS_PV(realisation="data")
    dataset_mock = PowerSpectrum_SDSS_PV(realisation=None)
    dataset_mock_red = PowerSpectrum_SDSS_PV(realisation=None, reduce_cov_factor=np.sqrt(2048.0))
    model = PowerBeutler2017(
        recon=dataset_data.recon,
        isotropic=dataset_data.isotropic,
        marg="full",
        correction=Correction.NONE,
        fix_params=["om", "sigma_s", "sigma_nl"],
        n_poly=1,
        dilate_smooth=False,
    )
    model.set_default("sigma_s", 0.0)
    model.set_default("sigma_nl", 9.0)
    # model.sanity_check(dataset_mock_red)
    model.sanity_check(dataset_mock)
    model.sanity_check(dataset_data)
