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
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
        dilate_smooth=True,
        n_poly=3,
    ):

        self.n_poly = n_poly
        self.dilate_smooth = dilate_smooth

        self.recon_smoothing_scale = None
        if isotropic:
            poly_poles = [0]
        if marg is not None:
            fix_params = list(fix_params)
            for pole in poly_poles:
                fix_params.extend([f"a{{{pole}}}_1_{{{1}}}", f"a{{{pole}}}_2_{{{1}}}", f"a{{{pole}}}_3_{{{1}}}"])
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
        if self.marg:
            for pole in self.poly_poles:
                self.set_default(f"a{{{pole}}}_1_{{{1}}}", 0.0)
                self.set_default(f"a{{{pole}}}_2_{{{1}}}", 0.0)
                self.set_default(f"a{{{pole}}}_3_{{{1}}}", 0.0)

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
            self.add_param(f"a{{{pole}}}_1_{{{1}}}", f"$a_{{{pole},1,1}}$", -100.0, 100.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param(f"a{{{pole}}}_2_{{{1}}}", f"$a_{{{pole},2,1}}$", -2.0, 2.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param(f"a{{{pole}}}_3_{{{1}}}", f"$a_{{{pole},3,1}}$", -0.2, 0.2, 0)  # Monopole Polynomial marginalisation 3

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
        xi_comp = [self.pk2xi_0.__call__(ks, pks[0], dist), np.zeros(len(dist)), np.zeros(len(dist))]

        if not self.isotropic:
            xi_comp[1] = self.pk2xi_2.__call__(ks, pks[2], dist)
            xi_comp[2] = self.pk2xi_4.__call__(ks, pks[4], dist)

        xi, poly = self.add_three_poly(dist, p, xi_comp)

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
        mocktype="abacus_cubicbox",
        redshift_bin=0,
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
        correction=Correction.HARTLAP,
    )
    model.set_default("sigma_s", 0.0)
    model.set_default("sigma_nl_perp", 4.0)
    model.set_default("sigma_nl_par", 8.0)
    model.sanity_check(dataset)

    """model.set_data(data)
    ks = model.camb.ks

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    # This function returns the values of k'(k,mu), pk (technically only the unmarginalised terms)
    # and model components for analytically marginalised parameters
    p = model.get_param_dict(model.get_defaults())
    p["alpha"] = 1.1
    p["epsilon"] = 0.05
    p["sigma_nl_perp"] = 4.0
    p["sigma_nl_par"] = 8.0
    p["beta"] = 0.5
    p["b{0}_{1}"] = 4.0
    print(model.get_alphas(p["alpha"], p["epsilon"]), p)
    newdist = np.linspace(1.0, 200.0, 2000)
    sprime, xi, marged = model.compute_correlation_function(newdist, p)
    print(sprime, xi)
    np.savetxt("../../barry/data/desi_kp4/test_xi_reasonable.dat", np.c_[newdist, xi[0], xi[1], xi[2]])

    xiells = np.loadtxt("../../barry/data/desi_kp4/xiells.txt").T

    import matplotlib.pyplot as plt

    vfac = 1.0 / (1.2127500000000002 * 1.0476190476190477**2)
    print(vfac)

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[1])
    plt.errorbar(sprime, sprime**2 * xi[0] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[2])
    plt.errorbar(sprime, sprime**2 * xi[1] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[3])
    plt.errorbar(sprime, sprime**2 * xi[2] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()"""
