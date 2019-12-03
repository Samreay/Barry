from barry.datasets.dataset import MultiDataset
from barry.datasets.dataset_power_spectrum_abc import PowerSpectrum


class PowerSpectrum_SDSS_DR12_Z061_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.61    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
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
        )


class PowerSpectrum_SDSS_DR12_Z051_NGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for NGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
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
        )


class PowerSpectrum_SDSS_DR12_Z051_SGC(PowerSpectrum):
    """ Power spectrum for SDSS BOSS DR12 sample for SGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
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
        )


class PowerSpectrum_SDSS_DR12_Z051(MultiDataset):
    """ Power spectrum for SDSS BOSS DR12 sample for combined NGC and SGC with mean redshift z = 0.51    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=1, postprocess=None):
        ngc = PowerSpectrum_SDSS_DR12_Z051_NGC(
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
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
        )
        super().__init__(name, [ngc, sgc])


class PowerSpectrum_SDSS_DR7_Z015(PowerSpectrum):
    """ Power spectrum for SDSS MGS DR7 sample for with mean redshift z = 0.15    """

    def __init__(self, realisation=None, name=None, fake_diag=False, recon=True, min_k=0.02, max_k=0.3, reduce_cov_factor=1, step_size=5, postprocess=None):
        super().__init__(
            "sdss_dr7_pk.pkl",
            name=name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            recon=recon,
            reduce_cov_factor=reduce_cov_factor,
            postprocess=postprocess,
            realisation=realisation,
            fake_diag=fake_diag,
        )
