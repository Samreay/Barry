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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    isotropic = False
    # datasets = [
    #    CorrelationFunction_ROSS_DR12_Z038(isotropic=isotropic, realisation="data"),
    #    CorrelationFunction_ROSS_DR12_Z051(isotropic=isotropic, realisation="data"),
    #    CorrelationFunction_ROSS_DR12_Z061(isotropic=isotropic, realisation="data"),
    # ]
    datasets = [CorrelationFunction_DESIMockChallenge(isotropic=isotropic, recon=True)]
    for dataset in datasets:
        for i in range(7):
            dataset.set_realisation(i)
            data = dataset.get_data()
            label = [r"$\xi_{0}(s)$", r"$\xi_{2}(s)$"] if i == 0 else [None, None]
            fmt = "o" if i == 0 else "None"
            ls = "None" if i == 0 else "-"
            if isotropic:
                yerr = (
                    data[0]["dist"] * np.sqrt(np.diag(data[0]["cov"]))[: len(data[0]["dist"])] if i == 0 else np.zeros(len(data[0]["dist"]))
                )
                plt.errorbar(
                    data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi0"], yerr=yerr, marker=fmt, ls=ls, c="r", zorder=i, label=label[0]
                )
            else:
                yerr = (
                    [
                        data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[: len(data[0]["dist"])],
                        data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[len(data[0]["dist"]) : 2 * len(data[0]["dist"])],
                    ]
                    if i == 0
                    else [None, None]
                )
                plt.errorbar(
                    data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi0"], yerr=yerr[0], marker=fmt, ls=ls, c="r", zorder=i, label=label[0]
                )
                plt.errorbar(
                    data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi2"], yerr=yerr[1], marker=fmt, ls=ls, c="b", zorder=i, label=label[1]
                )
        plt.xlabel(r"$s$")
        plt.ylabel(r"$s^{2}\xi_{\ell}(s)$")
        plt.title(dataset.name)
        plt.legend()
        plt.show()
