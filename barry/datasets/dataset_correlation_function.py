import logging

from barry.datasets.dataset_correlation_function_abc import CorrelationFunction


class CorrelationFunction_SDSS_DR12_Z061_NGC(CorrelationFunction):
    """ Correlation function for SDSS BOSS DR12 sample for the NGC with mean redshift z = 0.61    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):
        super().__init__(
            "sdss_dr12_z061_corr_ngc.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            realisation=realisation,
            isotropic=isotropic,
        )


class CorrelationFunction_ROSS_DR12_Z038(CorrelationFunction):
    """ Anisotropic Correlation function for SDSS BOSS DR12 sample from Ross 2017 with mean redshift z = 0.38    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for ROSS_DR12_Z038")
        super().__init__(
            "ross_2016_dr12_combined_corr_zbin0p38.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=1000,
            realisation=realisation,
            isotropic=isotropic,
        )


class CorrelationFunction_ROSS_DR12_Z051(CorrelationFunction):
    """ Anisotropic Correlation function for SDSS BOSS DR12 sample from Ross 2017 with mean redshift z = 0.51    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for ROSS_DR12_Z051")
        super().__init__(
            "ross_2016_dr12_combined_corr_zbin0p51.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=1000,
            realisation=realisation,
            isotropic=isotropic,
        )


class CorrelationFunction_ROSS_DR12_Z061(CorrelationFunction):
    """ Anisotropic Correlation function for SDSS BOSS DR12 sample from Ross 2017 with mean redshift z = 0.61    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for ROSS_DR12_Z061")
        super().__init__(
            "ross_2016_dr12_combined_corr_zbin0p61.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=1000,
            realisation=realisation,
            isotropic=isotropic,
        )


class CorrelationFunction_ROSS_DR12_Z061(CorrelationFunction):
    """ Anisotropic Correlation function for SDSS BOSS DR12 sample from Ross 2017 with mean redshift z = 0.61    """

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for ROSS_DR12_Z061")
        super().__init__(
            "ross_2016_dr12_combined_corr_zbin0p61.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=1000,
            realisation=realisation,
            isotropic=isotropic,
        )


class CorrelationFunction_DESIMockChallenge(CorrelationFunction):
    """ Power spectrum from the DESI Mock Challenge  """

    def __init__(
        self,
        name=None,
        min_dist=30,
        max_dist=200,
        recon=True,
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
        fit_poles=(0, 2, 4),
    ):
        if not recon:
            raise NotImplementedError("Pre-recon data not available for DESI Mock Challenge")
        super().__init__(
            "desi_mock_challenge_stage_2_xi.pkl",
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
            fit_poles=fit_poles,
        )


class CorrelationFunction_DESIMockChallenge_Post(CorrelationFunction):
    """ Power spectrum from the DESI Mock Challenge  """

    def __init__(
        self,
        name=None,
        min_dist=30,
        max_dist=200,
        recon=True,
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation=None,
        isotropic=False,
        fit_poles=(0, 2, 4),
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
            "desi_mock_challenge_post_stage_2_xi_pre_std.pkl",
            "desi_mock_challenge_post_stage_2_xi_pre_fix.pkl",
            "desi_mock_challenge_post_stage_2_xi_iso_5.pkl",
            "desi_mock_challenge_post_stage_2_xi_iso_10.pkl",
            "desi_mock_challenge_post_stage_2_xi_iso_15.pkl",
            "desi_mock_challenge_post_stage_2_xi_iso_20.pkl",
            "desi_mock_challenge_post_stage_2_xi_ani_5.pkl",
            "desi_mock_challenge_post_stage_2_xi_ani_10.pkl",
            "desi_mock_challenge_post_stage_2_xi_ani_15.pkl",
            "desi_mock_challenge_post_stage_2_xi_ani_20.pkl",
        ]
        datafile = datafiles[types.index(type.lower())]

        super().__init__(
            datafile,
            name=name,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
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

    # Plot the data and mock average for the desi_mock_challenge_post_stage_2 spectra
    isotropic = False
    types = ["rec-iso15", "rec-ani15"]
    recons = [True, True]
    realisations = [1, 1]
    for j, type in enumerate(types):
        datasets = [
            CorrelationFunction_DESIMockChallenge_Post(
                isotropic=isotropic, recon=recons[j], fit_poles=[0, 2, 4], min_dist=2, max_dist=205, type=type
            )
        ]
        for dataset in datasets:
            for i in range(realisations[j]):
                dataset.set_realisation(i)
                data = dataset.get_data()
                label = [r"$\xi_{0}(s)$", r"$\xi_{2}(s)$", r"$\xi_{4}(s)$"] if i == 0 else [None, None, None]
                fmt = "o" if i == 0 else "None"
                ls = "None" if i == 0 else "-"
                if isotropic:
                    yerr = (
                        data[0]["dist"] * np.sqrt(np.diag(data[0]["cov"]))[: len(data[0]["dist"])]
                        if i == 0
                        else np.zeros(len(data[0]["dist"]))
                    )
                    plt.errorbar(
                        data[0]["dist"],
                        data[0]["dist"] ** 2 * data[0]["xi0"],
                        yerr=yerr,
                        marker=fmt,
                        ls=ls,
                        c="r",
                        zorder=i,
                        label=label[0],
                    )
                else:
                    yerr = (
                        [
                            data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[: len(data[0]["dist"])],
                            data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[len(data[0]["dist"]) : 2 * len(data[0]["dist"])],
                            data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[2 * len(data[0]["dist"]) :],
                        ]
                        if i == 0
                        else [None, None, None]
                    )
                    print(yerr)
                    plt.errorbar(
                        data[0]["dist"],
                        data[0]["dist"] ** 2 * data[0]["xi0"],
                        yerr=yerr[0],
                        marker=fmt,
                        ls=ls,
                        c="r",
                        zorder=i,
                        label=label[0],
                    )
                    plt.errorbar(
                        data[0]["dist"],
                        data[0]["dist"] ** 2 * data[0]["xi2"],
                        yerr=yerr[1],
                        marker=fmt,
                        ls=ls,
                        c="b",
                        zorder=i,
                        label=label[1],
                    )
                    plt.errorbar(
                        data[0]["dist"],
                        data[0]["dist"] ** 2 * data[0]["xi4"],
                        yerr=yerr[2],
                        marker=fmt,
                        ls=ls,
                        c="g",
                        zorder=i,
                        label=label[2],
                    )
            plt.xlabel(r"$s$")
            plt.ylabel(r"$s^{2}\xi_{\ell}(s)$")
            plt.title(dataset.name)
            plt.legend()
            plt.show()
