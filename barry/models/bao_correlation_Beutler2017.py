import logging
import sys

sys.path.append("../..")

from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
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
        dilate_smooth=True,
        n_poly=3,
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
            includeb2=False,
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
            n_poly=n_poly,
        )

        self.set_marg(fix_params, poly_poles, n_poly, do_bias=False)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("b{0}_{1}", r"$b{0}_{1}$", 0.1, 10.0, 1.0)  # Galaxy bias
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.0, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.0, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.0, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for pole in self.poly_poles:
            for ip in range(self.n_poly):
                self.add_param(f"a{{{pole}}}_{{{ip+1}}}_{{{1}}}", f"$a_{{{pole},{ip+1},1}}$", -100.0, 100.0, 0)

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
            and 3 bias parameters and polynomial terms per multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        ks, pks, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, nopoly=True)
        xi_comp = np.array([self.pk2xi_0.__call__(ks, pks[0], dist), np.zeros(len(dist)), np.zeros(len(dist))])

        if not self.isotropic:
            xi_comp[1] = self.pk2xi_2.__call__(ks, pks[2], dist)
            xi_comp[2] = self.pk2xi_4.__call__(ks, pks[4], dist)

        xi, poly = self.add_poly(dist, p, xi_comp)

        return dist, xi, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import (
        CorrelationFunction_ROSS_DR12,
        CorrelationFunction_DESIMockChallenge_Post,
        CorrelationFunction_DESI_KP4,
    )
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

    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om", "sigma_s", "sigma_nl_par", "sigma_nl_perp"],
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=3,
    )
    model.set_default("sigma_s", 0.0)
    model.set_default("sigma_nl_perp", 1.8)
    model.set_default("sigma_nl_par", 5.4)
    print(model.get_active_params())

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
