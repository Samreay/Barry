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

        self.dilate_smooth = dilate_smooth

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
        min_dist=0.0,
        max_dist=200.0,
        mocktype="abacus_cubicbox",
        redshift_bin=0,
        num_mocks=1000,
        reduce_cov_factor=1,
    )
    data = dataset.get_data()
    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        mocktype="abacus_cubicbox",
        redshift_bin=0,
        num_mocks=1000,
        reduce_cov_factor=1,
    )
    fitdata = dataset.get_data()

    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om", "sigma_s", "sigma_nl_par", "sigma_nl_perp"],
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=1,
    )
    model.set_default("sigma_s", 0.0)
    model.set_default("sigma_nl_perp", 2.5)
    model.set_default("sigma_nl_par", 4.0)

    model.set_data(data)
    ks = model.camb.ks

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
    exit()

    # This function returns the values of k'(k,mu), pk (technically only the unmarginalised terms)
    # and model components for analytically marginalised parameters
    p = model.get_param_dict(model.get_defaults())
    p["sigma_nl_perp"] = 2.5
    p["sigma_nl_par"] = 4.0
    p["sigma_s"] = 0.0

    # Stephen's values
    p["alpha"], p["epsilon"] = model.get_reverse_alphas(0.99954812, 0.9993866)
    p["beta"] = 0.35241486 / (1.0 + 0.39695846)
    p["b{0}_{1}"] = (1.0 + 0.39695846) ** 2
    # p["a{0}_1_{1}"], p["a{0}_2_{1}"], p["a{0}_3_{1}"] = np.array([-1.05262736, 2.28522830e-03, 2.06412486e-04])
    # p["a{2}_1_{1}"], p["a{2}_2_{1}"], p["a{2}_3_{1}"] = np.array([-5.56598875, 1.39743506e-01, -2.98248166e-03])
    p["a{0}_1_{1}"], p["a{0}_2_{1}"], p["a{0}_3_{1}"] = np.array([2.06412486, 2.28522830e-03, -1.05262736e-04])
    p["a{2}_1_{1}"], p["a{2}_2_{1}"], p["a{2}_3_{1}"] = np.array([-2.98248166e01, 1.39743506e-01, -5.56598875e-04])
    print(model.get_alphas(p["alpha"], p["epsilon"]), p)
    print(data[0]["dist_input"])
    xi_model, poly_model = model.get_model(p, data[0])
    print(-2.0 * model.get_likelihood(p, fitdata[0]))

    # My values
    p["alpha"], p["epsilon"] = 0.9986549, -9.505569e-05
    p["beta"] = 0.30220875
    p["b{0}_{1}"] = 1.913501
    p["a{0}_1_{1}"], p["a{0}_2_{1}"], p["a{0}_3_{1}"] = np.array([1.24148167, 7.57507639e-03, -1.01977078e-04])
    p["a{2}_1_{1}"], p["a{2}_2_{1}"], p["a{2}_3_{1}"] = np.array([-1.81682704e01, 8.93597169e-02, -4.26365299e-04])
    xi_model2, poly_model2 = model.get_model(p, data[0])
    print(-2.0 * model.get_likelihood(p, fitdata[0]))

    xiells = np.loadtxt("../../barry/data/desi_kp4/bestfit.txt").T

    import matplotlib.pyplot as plt

    plt.plot(xiells[0], 100.0 * (xiells[1] / xi_model[:50] - 1.0))
    # plt.errorbar(sprime, sprime**2 * xi[0])
    plt.ylim(-0.1, 0.1)
    plt.show()

    plt.plot(xiells[0], 100.0 * (xiells[2] / xi_model[50:100] - 1.0))
    # plt.errorbar(sprime, sprime**2 * xi[1])
    # plt.errorbar(data[0]["dist"], data[0]["dist"] ** 2 * data[0]["xi2"], ls=None, marker="o")
    plt.ylim(-0.1, 0.1)
    plt.show()
