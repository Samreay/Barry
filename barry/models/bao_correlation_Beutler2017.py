import sys

sys.path.append("../..")
from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
import numpy as np


class CorrBeutler2017(CorrelationFunctionFit):
    """xi(s) model inspired from Beutler 2017 that treats alphas in the same way as P(k)."""

    def __init__(
        self,
        name="Corr Beutler 2017",
        fix_params=("om",),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        dilate_smooth=False,
        n_poly=(0, 2),
    ):

        self.dilate_smooth = dilate_smooth

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            n_poly=n_poly,
        )
        self.parent = PowerBeutler2017(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            dilate_smooth=dilate_smooth,
            broadband_type=None,
        )

        self.set_marg(fix_params, poly_poles, n_poly, do_bias=True, marg_bias=1)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.0, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.0, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.0, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for pole in self.poly_poles:
            for ip in self.n_poly:
                self.add_param(f"a{{{pole}}}_{{{ip}}}_{{{1}}}", f"$a_{{{pole},{ip},1}}$", -10.0, 10.0, 0)


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import (
        CorrelationFunction_DESI_KP4,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2, 4],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    data = dataset.get_data()

    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=[0, 2],
    )
    model.set_default("sigma_nl_par", 5.4, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_nl_perp", 1.6, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
