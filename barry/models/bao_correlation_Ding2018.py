import sys

sys.path.append("../..")
from barry.models import PowerDing2018
from barry.models.bao_correlation import CorrelationFunctionFit
import numpy as np


class CorrDing2018(CorrelationFunctionFit):
    """xi(s) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(
        self,
        name="Corr Ding 2018",
        fix_params=("om", "beta"),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        include_binmat=True,
        broadband_type="spline",
        **kwargs,
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
            include_binmat=include_binmat,
            broadband_type=broadband_type,
            **kwargs,
        )
        self.parent = PowerDing2018(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            broadband_type=None,
        )

        self.set_marg(fix_params, do_bias=False)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", -5.0, 5.0, 0.0)  # Non-linear galaxy bias


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = CorrelationFunction_DESI_KP4(
        recon=None,
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    data = dataset.get_data()

    model = CorrDing2018(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om", "sigma_s"],
        poly_poles=dataset.fit_poles,
        correction=Correction.HARTLAP,
        n_poly=[0, 2],
    )
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
