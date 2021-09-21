import logging

import sys

sys.path.append("..")
from barry.datasets.dataset import MultiDataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum


class PowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_ngc_z3.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_SDSS_DR12_Z061_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.61    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_sgc_z3.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_SDSS_DR12_Z051_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.51    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_ngc_z2.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_SDSS_DR12_Z051_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.51    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_sgc_z2.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_SDSS_DR12_Z038_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.38    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_ngc_z1.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_SDSS_DR12_Z038_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.38    """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=2,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=True,
        fit_poles=None,
    ):
        super().__init__(
            "sdss_dr12_pk_sgc_z1.pkl",
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
            fit_poles=fit_poles,
        )


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
        fit_poles=None,
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
            fit_poles=fit_poles,
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
        fit_poles=None,
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
            fit_poles=fit_poles,
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
        fit_poles=None,
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
            fit_poles=fit_poles,
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
            fit_poles=fit_poles,
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
        fit_poles=None,
    ):

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
            fit_poles=fit_poles,
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
        fit_poles=None,
    ):

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
            fit_poles=fit_poles,
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
        fit_poles=None,
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
            fit_poles=fit_poles,
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
            fit_poles=fit_poles,
        )
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_DESIMockChallenge(PowerSpectrum):
    """ Power spectrum from the DESI Mock Challenge  """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=True,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
        fit_poles=(0, 2),
    ):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for DESIMockChallenge")

        if any(pole in [1, 3, 4] for pole in fit_poles):
            raise NotImplementedError("Only monopole and quadrupole included in DESIMockChallenge")

        super().__init__(
            "desi_mock_challenge_stage_2_pk.pkl",
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_DESIMockChallenge_Post(PowerSpectrum):
    """ Power spectrum from the DESI Mock Challenge  """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
        fit_poles=(0, 2),
        type="cov-std",
    ):

        types = ["cov-std", "cov-fix", "rec-iso5", "rec-iso10", "rec-iso15", "rec-iso20", "rec-ani5", "rec-ani10", "rec-ani15", "rec-ani20"]
        if type.lower() not in types:
            raise NotImplementedError("Type not recognised, must be cov-std, cov-fix, rec-iso5/10/15/20 or rec-ani5/10/15/20")

        if type.lower() in types[:2] and recon is True:
            raise NotImplementedError("Type corresponds to pre-recon, but recon=True")

        if type.lower() in types[2:] and recon is False:
            raise NotImplementedError("Type corresponds to post-recon, but recon=False")

        if any(pole in [1, 3] for pole in fit_poles):
            raise NotImplementedError("Only even multipoles included in DESIMockChallenge")

        datafiles = [
            "desi_mock_challenge_post_stage_2_pk_pre_std.pkl",
            "desi_mock_challenge_post_stage_2_pk_pre_fix.pkl",
            "desi_mock_challenge_post_stage_2_pk_iso_5.pkl",
            "desi_mock_challenge_post_stage_2_pk_iso_10.pkl",
            "desi_mock_challenge_post_stage_2_pk_iso_15.pkl",
            "desi_mock_challenge_post_stage_2_pk_iso_20.pkl",
            "desi_mock_challenge_post_stage_2_pk_ani_5.pkl",
            "desi_mock_challenge_post_stage_2_pk_ani_10.pkl",
            "desi_mock_challenge_post_stage_2_pk_ani_15.pkl",
            "desi_mock_challenge_post_stage_2_pk_ani_20.pkl",
        ]
        datafile = datafiles[types.index(type.lower())]

        super().__init__(
            datafile,
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
            fit_poles=fit_poles,
        )


class PowerSpectrum_DESILightcone_Mocks_Recon(PowerSpectrum):
    """ Power spectrum from the DESI Mock Challenge  """

    def __init__(
        self,
        name=None,
        min_k=0.02,
        max_k=0.30,
        step_size=None,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        postprocess=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
        fit_poles=(0, 2),
        type="julian_reciso",
    ):

        types = [
            "julian_reciso",
            "julian_recsym",
            "martin_reciso",
            "martin_recsym",
        ]
        if type.lower() not in types:
            raise NotImplementedError("Type not recognised, must be julian_reciso, julian_recsym, martin_reciso, martin_recsym")

        if any(pole in [1, 3] for pole in fit_poles):
            raise NotImplementedError("Only even multipoles included in DESIMockChallenge")

        datafiles = [
            "desi_lightcone_mocks_recon_julian_reciso.pkl",
            "desi_lightcone_mocks_recon_julian_recsym.pkl",
            "desi_lightcone_mocks_recon_martin_reciso.pkl",
            "desi_lightcone_mocks_recon_martin_recsym.pkl",
        ]
        datafile = datafiles[types.index(type.lower())]

        super().__init__(
            datafile,
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
            fit_poles=fit_poles,
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    """# Plot the data and mock average for the sdss spectra
    for j, recon in enumerate(["iso", None]):
        datasets = [
            PowerSpectrum_SDSS_DR12_Z038_NGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
            PowerSpectrum_SDSS_DR12_Z038_SGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
            PowerSpectrum_SDSS_DR12_Z051_NGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
            PowerSpectrum_SDSS_DR12_Z051_SGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
            PowerSpectrum_SDSS_DR12_Z061_NGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
            PowerSpectrum_SDSS_DR12_Z061_SGC(isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30),
        ]
        for dataset in datasets:
            for i, realisation in enumerate([None, "data", 500]):
                dataset.set_realisation(realisation)
                data = dataset.get_data()
                label = [r"$P_{0}(k)$", r"$P_{2}(k)$", r"$P_{4}(k)$"] if i == 0 else [None, None, None]
                fmt = "o" if i == 0 else "None"
                ls = "None" if i == 0 else "-"
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
            plt.show()"""

    # Plot the data and mock average for the sdss spectra
    for j, recon in enumerate(["iso", None]):
        datasets = [
            PowerSpectrum_DESILightcone_Mocks_Recon(
                isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30, type="julian_reciso"
            ),
            PowerSpectrum_DESILightcone_Mocks_Recon(
                isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30, type="julian_recsym"
            ),
            PowerSpectrum_DESILightcone_Mocks_Recon(
                isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30, type="martin_reciso"
            ),
            PowerSpectrum_DESILightcone_Mocks_Recon(
                isotropic=False, recon=recon, fit_poles=[0, 2], min_k=0.02, max_k=0.30, type="martin_recsym"
            ),
        ]
        for dataset in datasets:
            for i, realisation in enumerate([None, "data", 50]):
                dataset.set_realisation(realisation)
                data = dataset.get_data()
                label = [r"$P_{0}(k)$", r"$P_{2}(k)$", r"$P_{4}(k)$"] if i == 0 else [None, None, None]
                fmt = "o" if i == 0 else "None"
                ls = "None" if i == 0 else "-"
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
