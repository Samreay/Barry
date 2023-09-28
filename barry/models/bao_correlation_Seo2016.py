import sys

sys.path.append("../..")
import numpy as np
from barry.models import PowerSeo2016
from barry.models.bao_correlation import CorrelationFunctionFit


class CorrSeo2016(CorrelationFunctionFit):
    """xi(s) model inspired from Seo 2016.

    See https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.2453S for details.
    """

    def __init__(
        self,
        name="Corr Seo 2016",
        fix_params=("om", "beta"),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        n_poly=(0, 2),
    ):

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            includeb2=False,
            n_poly=n_poly,
        )
        self.parent = PowerSeo2016(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            broadband_type=None,
        )

        self.set_marg(fix_params, poly_poles, n_poly, do_bias=False)

    def declare_parameters(self):
        # Define parameters
        super().declare_parameters()
        self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
        self.add_param("sigma_s", r"$\Sigma_s$", 0.00, 10.0, 5.0)  # Fingers-of-god damping
        for pole in self.poly_poles:
            for ip in self.n_poly:
                self.add_param(f"a{{{pole}}}_{{{ip}}}_{{{1}}}", f"$a_{{{pole},{ip},1}}$", -10.0, 10.0, 0)


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    data = dataset.get_data()

    model = CorrSeo2016(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset.fit_poles,
        correction=Correction.HARTLAP,
        n_poly=[0, 2],
    )
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
