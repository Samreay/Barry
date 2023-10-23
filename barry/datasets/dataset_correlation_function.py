import logging

from barry.datasets.dataset_correlation_function_abc import CorrelationFunction


class CorrelationFunction_SDSS_DR12_Z061_NGC:
    """Correlation function for SDSS BOSS DR12 sample for the NGC with mean redshift z = 0.61"""

    def __init__(self, name=None, min_dist=30, max_dist=200, recon=True, reduce_cov_factor=1, realisation=None, isotropic=True):

        raise NotImplementedError("This class is currently not implemented. Will hopefully be reintegrated in the future.")

        """
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
        """


class CorrelationFunction_ROSS_DR12(CorrelationFunction):
    """Anisotropic Correlation function for SDSS BOSS DR12 sample from Ross 2017 with mean redshifts z = 0.38, 0.51 and 0.61.
    Only contains reconstructed data and a covariance matrix (no mocks and no pre-recon), for the monopole and quadrupole.
    """

    def __init__(
        self,
        redshift_bin=3,
        name=None,
        min_dist=30.0,
        max_dist=200.0,
        recon="iso",
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation="data",
        isotropic=True,
        fit_poles=(0,),
    ):

        self.nredshift_bins = 3
        self.nsmoothtypes = 1
        self.ndata = 1

        if recon.lower() != "iso":
            raise NotImplementedError("Only isotropic recon data not available for ROSS_DR12")

        if any(pole in [1, 3, 4] for pole in fit_poles):
            raise NotImplementedError("Only monopole and quadrupole included in ROSS_DR12")

        if realisation is not None:
            if isinstance(realisation, int):
                raise NotImplementedError("Only data (no mocks) available for ROSS_DR12")
            elif realisation.lower() != "data":
                raise ValueError("Realisation is set to a string, but not 'data'")
        else:
            raise NotImplementedError("Only data (no mocks) available for ROSS_DR12")

        if redshift_bin not in [1, 2, 3]:
            raise NotImplementedError("Redshift bin for ROSS_DR12 must be 1, 2 or 3, corresponding to 0.38, 0.51 and 0.61 respectively")

        reds = ["zbin0p38", "zbin0p51", "zbin0p61"]
        datafile = "ross_2016_dr12_combined_corr_" + reds[redshift_bin - 1] + ".pkl"

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


class CorrelationFunction_eBOSS_LRGpCMASS(CorrelationFunction):

    """Correlation Function for SDSS DR16 eBOSS LRGpCMASS sample for combined NGC and SGC at redshift 0.698."""

    def __init__(
        self,
        galactic_cap="both",
        name=None,
        min_dist=30.0,
        max_dist=200.0,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation="data",
        isotropic=True,
        fit_poles=(0,),
    ):

        if galactic_cap.lower() not in ["both"]:
            raise NotImplementedError("Separate NGC and SGC Galactic caps for eBOSS_LRGpCMASS Xi not available")

        if realisation is not None:
            if isinstance(realisation, int):
                raise NotImplementedError("Only data (no mocks) available for eBOSS LRGpCMASS Xi")
            elif realisation.lower() != "data":
                raise ValueError("Realisation is set to a string, but not 'data'")
        else:
            raise NotImplementedError("Only data (no mocks) available for eBOSS LRGpCMASS Xi")

        datafile = "sdss_dr16_lrgpcmass_xi_" + galactic_cap.lower() + ".pkl"

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


class CorrelationFunction_DESIMockChallenge_Post(CorrelationFunction):
    """Power spectrum from the DESI Mock Challenge"""

    def __init__(
        self,
        name=None,
        min_dist=30.0,
        max_dist=200.0,
        recon=None,
        reduce_cov_factor=1,
        num_mocks=None,
        fake_diag=False,
        realisation="data",
        isotropic=True,
        fit_poles=(0,),
        covtype="nonfix",
        smoothtype=3,
        tracer="elg",
    ):

        self.nredshift_bins = 1
        self.nsmoothtypes = 3
        self.ndata = 1

        if covtype.lower() not in ["fix", "nonfix", "analytic"]:
            raise NotImplementedError("covtype not recognised, must be fix, nonfix or analytic")

        smoothscales = ["5", "10", "15", "20"]
        if smoothtype not in [1, 2, 3, 4]:
            raise NotImplementedError(
                "smoothscale not recognised, must be 1, 2, 3 or 4, corresponding to 5, 10, 15 or 20 (Mpc/h) respectively"
            )

        if tracer.lower() not in ["elg", "elghd", "elgmd", "elgld"]:
            raise NotImplementedError("tracer not recognised, must be elg, elghd, elgmd, or elgld")

        if any(pole in [1, 3] for pole in fit_poles):
            raise NotImplementedError("Only even multipoles included in DESIMockChallenge")

        reconname = "iso" if recon is None else recon.lower()
        datafile = (
            "desi_mock_challenge_post_stage_2_xi_" + reconname + "_" + smoothscales[smoothtype - 1] + "_" + covtype + "_" + tracer + ".pkl"
        )

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


class CorrelationFunction_DESI_KP4(CorrelationFunction):
    """Correlation Function for DESI KP4"""

    def __init__(
        self,
        isotropic=False,
        min_dist=30.0,
        max_dist=200.0,
        recon=None,
        fake_diag=False,
        reduce_cov_factor=1,
        realisation=None,
        num_mocks=1000,
        fit_poles=(0,),
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
        data_location=None,
    ):

        self.nredshift_bins = 1
        self.nsmoothtypes = 1
        self.ndata = 1

        super().__init__(
            datafile,
            name=None,
            min_dist=min_dist,
            max_dist=max_dist,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            num_mocks=num_mocks,
            fake_diag=fake_diag,
            realisation=realisation,
            isotropic=isotropic,
            fit_poles=fit_poles,
            data_location=data_location,
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    if False:

        # Plot the data for the ROSS DR12 Correlation function
        for j, recon in enumerate(["iso"]):
            for redshift_bin in [1, 2, 3]:
                dataset = CorrelationFunction_ROSS_DR12(
                    redshift_bin=redshift_bin,
                    isotropic=False,
                    recon=recon,
                    realisation="data",
                    fit_poles=[0, 2],
                    min_dist=0.0,
                    max_dist=200.0,
                )
                data = dataset.get_data()
                label = [r"$\xi_{0}(s)$", r"$\xi_{2}(s)$", r"$\xi_{4}(s)$"]
                color = ["r", "b", "g"]
                fmt = "o"
                ls = "None"
                yerr = [
                    data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[: len(data[0]["dist"])],
                    data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[len(data[0]["dist"]) : 2 * len(data[0]["dist"])],
                    data[0]["dist"] ** 2 * np.sqrt(np.diag(data[0]["cov"]))[2 * len(data[0]["dist"]) :],
                ]
                for m, xi in enumerate(["xi0", "xi2", "xi4"]):
                    plt.errorbar(
                        data[0]["dist"],
                        data[0]["dist"] ** 2 * data[0][xi],
                        yerr=yerr[m],
                        marker=fmt,
                        ls=ls,
                        c=color[m],
                        label=label[m],
                    )
                plt.xlabel(r"$s$")
                plt.ylabel(r"$s^{2}\xi_{\ell}(s)$")
                plt.title(dataset.name)
                plt.legend()
                plt.show()
