import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge0_Z01
    from barry.models import PowerBeutler2017
    from barry.models.model import Correction

    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model_smooth = PowerBeutler2017(recon=False, isotropic=False, smooth=True, correction=Correction.NONE)

    dataset = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data")
    data = dataset.get_data()

    model.sanity_check(dataset, figname="desi_mock0_optimised_bestfit.png", niter=1000, maxiter=1000)
    # print(likelihood)
    #
    # model.plot_default(dataset)
