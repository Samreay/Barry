import logging

from barry.datasets.dataset import MultiDataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum


class PowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.61    """

    def __init__(
        self,
        realisation=None,
        name=None,
        fake_diag=False,
        recon=True,
        min_k=0.02,
        max_k=0.3,
        reduce_cov_factor=1,
        step_size=1,
        postprocess=None,
        isotropic=True,
    ):
        super().__init__(
            "sdss_dr12_z061_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
            isotropic=isotropic,
        )


class PowerSpectrum_SDSS_DR12_Z051_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.51    """

    def __init__(
        self,
        realisation=None,
        name=None,
        fake_diag=False,
        recon=True,
        min_k=0.02,
        max_k=0.3,
        reduce_cov_factor=1,
        step_size=1,
        postprocess=None,
        isotropic=True,
    ):
        super().__init__(
            "sdss_dr12_z051_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
            isotropic=isotropic,
        )


class PowerSpectrum_SDSS_DR12_Z051_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.51    """

    def __init__(
        self,
        realisation=None,
        name=None,
        fake_diag=False,
        recon=True,
        min_k=0.02,
        max_k=0.3,
        reduce_cov_factor=1,
        step_size=1,
        postprocess=None,
        isotropic=True,
    ):
        super().__init__(
            "sdss_dr12_z051_pk_sgc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
            isotropic=isotropic,
        )


class PowerSpectrum_SDSS_DR12_Z051(MultiDataset):
    """ Power spectrum for SDSS BOSS DR12 sample for combined NGC and SGC with mean redshift z = 0.51    """

    def __init__(
        self,
        realisation=None,
        name=None,
        fake_diag=False,
        recon=True,
        min_k=0.02,
        max_k=0.3,
        reduce_cov_factor=1,
        step_size=1,
        postprocess=None,
        isotropic=True,
    ):
        ngc = PowerSpectrum_SDSS_DR12_Z051_NGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
            isotropic=isotropic,
        )
        sgc = PowerSpectrum_SDSS_DR12_Z051_SGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
            isotropic=isotropic,
        )
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_Beutler2019_Z038_NGC(PowerSpectrum):
    """ Power spectrum from Beutler 2019 for DR12 sample for NGC with mean redshift z = 0.38    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):

        if recon:
            raise NotImplementedError("Post-recon data not available for Beutler2019_DR12_Z038")

        super().__init__(
            "beutler_2019_dr12_z038_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


class PowerSpectrum_Beutler2019_Z038_SGC(PowerSpectrum):
    """ Power spectrum from Beutler 2019 for DR12 sample for SGC with mean redshift z = 0.38    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        if recon:
            raise NotImplementedError("Post-recon data not available for Beutler2019_DR12_Z038")

        super().__init__(
            "beutler_2019_dr12_z038_pk_sgc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


class PowerSpectrum_Beutler2019_Z038(MultiDataset):
    """ Power spectrum from Beutler 2019 for DR12 sample for combined NGC and SGC with mean redshift z = 0.38    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        ngc = PowerSpectrum_Beutler2019_Z038_NGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )
        sgc = PowerSpectrum_Beutler2019_Z038_SGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_Beutler2019_Z061_NGC(PowerSpectrum):
    """ Power spectrum from Beutler 2019 for DR12 sample for NGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        if recon:
            raise NotImplementedError("Post-recon data not available for Beutler2019_DR12_Z061")

        super().__init__(
            "beutler_2019_dr12_z061_pk_ngc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


class PowerSpectrum_Beutler2019_Z061_SGC(PowerSpectrum):
    """ Power spectrum from Beutler 2019 for DR12 sample for SGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        if recon:
            raise NotImplementedError("Post-recon data not available for Beutler2019_DR12_Z061")

        super().__init__(
            "beutler_2019_dr12_z061_pk_sgc.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


class PowerSpectrum_Beutler2019_Z061(MultiDataset):
    """ Power spectrum from Beutler 2019 for DR12 sample for combined NGC and SGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        ngc = PowerSpectrum_Beutler2019_Z061_NGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )
        sgc = PowerSpectrum_Beutler2019_Z061_SGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_DESIMockChallenge0_Z01(PowerSpectrum):
    """ Power spectrum from Beutler 2019 for DR12 sample for SGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
    ):
        if recon:
            raise NotImplementedError("Post-recon data not available for Beutler2019_DR12_Z061")

        super().__init__(
            "desi_mock_challenge_0.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


class PowerSpectrum_DESIMockChallenge_Handshake(PowerSpectrum):
    """ Power spectrum from the DESI Mock Challenge Handshake    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=False,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
    ):
        if recon:
            raise NotImplementedError("Post-recon data not available for DESIMockChallenge_Handshake")

        super().__init__(
            "desi_mock_challenge_handshake.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            postprocess=postprocess,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Plot the data and mock average for the Beutler 2019 spectra
    for isotropic in [False]:
        datasets = [
            PowerSpectrum_Beutler2019_Z038_NGC(isotropic=isotropic),
            PowerSpectrum_Beutler2019_Z038_SGC(isotropic=isotropic),
            PowerSpectrum_Beutler2019_Z061_NGC(isotropic=isotropic),
            PowerSpectrum_Beutler2019_Z061_SGC(isotropic=isotropic),
        ]
        for dataset in datasets:
            for i, realisation in enumerate([None, "data"]):
                dataset.set_realisation(realisation)
                data = dataset.get_data()
                label = [r"$P_{0}(k)$", r"$P_{2}(k)$", r"$P_{4}(k)$"] if i == 0 else [None, None, None]
                fmt = "o" if i == 0 else "None"
                ls = "None" if i == 0 else "-"
                if isotropic:
                    yerr = data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"])) if i == 0 else np.zeros(len(data[0]["ks"]))
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk0"], yerr=yerr, marker=fmt, ls=ls, c="r", zorder=i, label=label[0]
                    )
                else:
                    yerr = (
                        [
                            data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"]))[0 : len(data[0]["ks"])],
                            data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"]))[2 * len(data[0]["ks"]) : 3 * len(data[0]["ks"])],
                            data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"]))[4 * len(data[0]["ks"]) : 5 * len(data[0]["ks"])],
                        ]
                        if i == 0
                        else [None, None, None]
                    )
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk0"], yerr=yerr[0], marker=fmt, ls=ls, c="r", zorder=i, label=label[0]
                    )
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk2"], yerr=yerr[1], marker=fmt, ls=ls, c="b", zorder=i, label=label[1]
                    )
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk4"], yerr=yerr[2], marker=fmt, ls=ls, c="g", zorder=i, label=label[2]
                    )
            plt.xlabel(r"$k$")
            plt.ylabel(r"$k\,P(k)$")
            plt.title(dataset.name)
            plt.legend()
            plt.show()

            if not isotropic:
                for i, realisation in enumerate([None, "data"]):
                    dataset.set_realisation(realisation)
                    data = dataset.get_data()
                    label = [r"$P_{1}(k)$", r"$P_{3}(k)$"] if i == 0 else [None, None]
                    fmt = "o" if i == 0 else "None"
                    ls = "None" if i == 0 else "-"
                    yerr = (
                        [
                            data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"]))[1 * len(data[0]["ks"]) : 2 * len(data[0]["ks"])],
                            data[0]["ks"] * np.sqrt(np.diag(data[0]["cov"]))[3 * len(data[0]["ks"]) : 4 * len(data[0]["ks"])],
                        ]
                        if i == 0
                        else [None, None]
                    )
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk1"], yerr=yerr[0], marker=fmt, ls=ls, c="r", zorder=1, label=label[0]
                    )
                    plt.errorbar(
                        data[0]["ks"], data[0]["ks"] * data[0]["pk3"], yerr=yerr[1], marker=fmt, ls=ls, c="b", zorder=1, label=label[1]
                    )
                plt.xlabel(r"$k$")
                plt.ylabel(r"$k\,Im[P(k)]$")
                plt.title(dataset.name)
                plt.show()
