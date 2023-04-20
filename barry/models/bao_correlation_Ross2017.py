import logging
import sys

sys.path.append("../..")

from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
import numpy as np


class CorrRoss2017(CorrelationFunctionFit):
    """xi(s) model inspired from Beutler 2017 and Ross 2017."""

    def __init__(
        self,
        name="Corr Ross 2017",
        fix_params=("om",),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        includeb2=True,
        n_poly=3,
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
            includeb2=includeb2,
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
            n_poly=n_poly,
        )

        self.set_marg(fix_params, poly_poles, n_poly, do_bias=True)

        # We might not need to evaluate the hankel transform everytime, so check.
        if self.isotropic:
            if all(elem in self.fix_params for elem in ["sigma_s", "sigma_nl"]):
                self.fixed_xi = True
        else:
            if all(elem in self.fix_params for elem in ["sigma_s", "sigma_nl_perp", "sigma_nl_par", "beta"]):
                self.fixed_xi = True

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
            for pole in self.poly_poles:
                self.add_param(f"b{{{pole}}}_{{{1}}}", f"$b{{{pole}}}_{{{1}}}$", 0.01, 10.0, 1.0)  # Linear galaxy bias for each multipole
                for ip in range(self.n_poly):
                    self.add_param(f"a{{{pole}}}_{{{ip + 1}}}_{{{1}}}", f"$a_{{{pole},{ip + 1},1}}$", -100.0, 100.0, 0)

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
        sprime, xi_comp = self.compute_basic_correlation_function(dist, p, smooth=smooth)
        xi, poly = self.add_poly(dist, p, xi_comp)

        return sprime, xi, poly

    def add_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but with polynomial terms for each multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = p["b{0}_{1}"] * xi0
            poly = np.zeros((1, len(dist)))
            if self.marg:
                poly = [1.0 / dist ** (ip - 1) for ip in range(self.n_poly)]
            else:
                for ip in range(self.n_poly):
                    xi[0] += p[f"a{0}_{{{ip+1}}}_{1}"] / (dist ** (ip - 1))

        else:
            xi[0] = p["b{0}_{1}"] * xi0
            if self.includeb2:
                xi[1] = 2.5 * (p["b{2}_{1}"] * xi2 - xi[0])
                if 4 in self.poly_poles:
                    xi[2] = 1.125 * (p["b{4}_{1}"] * xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
                else:
                    xi[2] = 1.125 * (xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
            else:
                xi[1] = 2.5 * p["b{0}_{1}"] * (xi2 - xi0)
                xi[2] = 1.125 * p["b{0}_{1}"] * (xi4 - 10.0 * xi2 + 3.0 * xi0)

            # Polynomial shape
            if self.marg:
                if self.includeb2:
                    xi_marg = [xi0, 2.5 * xi2, 1.125 * xi4]
                    poly = np.zeros((4 * len(self.poly_poles), 3, len(dist)))
                    for npole, pole in enumerate(self.poly_poles):
                        poly[(self.n_poly + 1) * npole, npole] = xi_marg[npole]
                        poly[(self.n_poly + 1) * npole + 1 : (self.n_poly + 1) * (npole + 1), npole] = [
                            1.0 / dist ** (ip - 1) for ip in range(self.n_poly)
                        ]
                    poly[0, 1] = -2.5 * xi0
                    poly[0, 2] = 1.125 * 3.0 * xi0
                    if 2 in self.poly_poles:
                        poly[4, 2] = -1.125 * 10.0 * xi2

                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
                else:
                    poly = np.zeros((3 * len(self.poly_poles) + 1, 3, len(dist)))
                    poly[0] = [xi0, 2.5 * (xi2 - xi0), 1.125 * (xi4 - 10.0 * xi2 + 3.0 * xi0)]
                    for npole, pole in enumerate(self.poly_poles):
                        poly[self.n_poly * npole + 1 : self.n_poly * (npole + 1) + 1, npole] = [
                            1.0 / dist ** (ip - 1) for ip in range(self.n_poly)
                        ]
                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
            else:
                poly = np.zeros((1, 3, len(dist)))
                for pole in self.poly_poles:
                    for ip in range(self.n_poly):
                        xi[int(pole / 2)] += p[f"a{{{pole}}}_{{{ip + 1}}}_{1}"] / dist ** (ip - 1)

        return xi, poly

    def add_zero_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but without any polynomial terms

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = p["b{0}_{1}"] * xi0
            poly = np.zeros((1, len(dist)))
        else:
            xi[0] = p["b{0}_{1}"] * xi0
            if self.includeb2:
                xi[1] = 2.5 * (p["b{2}_{1}"] * xi2 - xi[0])
                if 4 in self.poly_poles:
                    xi[2] = 1.125 * (p["b{4}_{1}"] * xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
                else:
                    xi[2] = 1.125 * (xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
            else:
                xi[1] = 2.5 * p["b{0}_{1}"] * (xi2 - xi0)
                xi[2] = 1.125 * p["b{0}_{1}"] * (xi4 - 10.0 * xi2 + 3.0 * xi0)

            # Polynomial shape
            if self.marg:
                if self.includeb2:
                    xi_marg = [xi0, 2.5 * xi2, 1.125 * xi4]
                    poly = np.zeros((len(self.poly_poles), 3, len(dist)))
                    for npole, pole in enumerate(self.poly_poles):
                        poly[npole, npole] = [xi_marg[npole]]
                    poly[0, 1] = -2.5 * xi0
                    poly[0, 2] = 1.125 * 3.0 * xi0
                    if 2 in self.poly_poles:
                        poly[1, 2] = -1.125 * 10.0 * xi2

                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
                else:
                    poly = np.zeros((1, 3, len(dist)))
                    poly[0] = [xi0, 2.5 * (xi2 - xi0), 1.125 * (xi4 - 10.0 * xi2 + 3.0 * xi0)]

                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
            else:
                poly = np.zeros((1, 3, len(dist)))

        return xi, poly

    def add_three_poly(self, dist, p, xi_comp):
        """Converts the xi components to a full model but with 3 polynomial terms for each multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        xi_comp : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the convert model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        xi0, xi2, xi4 = xi_comp
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            xi[0] = p["b{0}_{1}"] * xi0
            poly = np.zeros((1, len(dist)))
            if self.marg:
                poly = [xi[0], 1.0 / (dist**2), 1.0 / dist, np.ones(len(dist))]
            else:
                xi[0] += p["a{0}_1_{1}"] / (dist**2) + p["a{0}_2_{1}"] / dist + p["a{0}_3_{1}"]

        else:
            xi[0] = p["b{0}_{1}"] * xi0
            if self.includeb2:
                xi[1] = 2.5 * (p["b{2}_{1}"] * xi2 - xi[0])
                if 4 in self.poly_poles:
                    xi[2] = 1.125 * (p["b{4}_{1}"] * xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
                else:
                    xi[2] = 1.125 * (xi4 - 10.0 * p["b{2}_{1}"] * xi2 + 3.0 * p["b{0}_{1}"] * xi0)
            else:
                xi[1] = 2.5 * p["b{0}_{1}"] * (xi2 - xi0)
                xi[2] = 1.125 * p["b{0}_{1}"] * (xi4 - 10.0 * xi2 + 3.0 * xi0)

            # Polynomial shape
            if self.marg:
                if self.includeb2:
                    xi_marg = [xi0, 2.5 * xi2, 1.125 * xi4]
                    poly = np.zeros((4 * len(self.poly_poles), 3, len(dist)))
                    for npole, pole in enumerate(self.poly_poles):
                        poly[4 * npole : 4 * (npole + 1), npole] = [xi_marg[npole], 1.0 / (dist**2), 1.0 / dist, np.ones(len(dist))]
                    poly[0, 1] = -2.5 * xi0
                    poly[0, 2] = 1.125 * 3.0 * xi0
                    if 2 in self.poly_poles:
                        poly[4, 2] = -1.125 * 10.0 * xi2

                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
                else:
                    poly = np.zeros((3 * len(self.poly_poles) + 1, 3, len(dist)))
                    poly[0] = [xi0, 2.5 * (xi2 - xi0), 1.125 * (xi4 - 10.0 * xi2 + 3.0 * xi0)]
                    for npole, pole in enumerate(self.poly_poles):
                        poly[3 * npole + 1 : 3 * (npole + 1) + 1, npole] = [1.0 / (dist**2), 1.0 / dist, np.ones(len(dist))]

                    xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
            else:
                poly = np.zeros((1, 3, len(dist)))
                for pole in self.poly_poles:
                    xi[int(pole / 2)] += (
                        p[f"a{{{pole}}}_1_{{{1}}}"] / dist**2 + p[f"a{{{pole}}}_2_{{{1}}}"] / dist + p[f"a{{{pole}}}_3_{{{1}}}"]
                    )

        return xi, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import (
        CorrelationFunction_DESI_KP4,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    """print("Checking isotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=True, recon="iso", realisation="data")
    model = CorrBeutler2017(recon=dataset.recon, marg="full", isotropic=dataset.isotropic, correction=Correction.NONE)
    model.sanity_check(dataset)

    print("Checking anisotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2], realisation="data")
    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.NONE,
    )
    model.sanity_check(dataset)"""

    print("Checking anisotropic data")
    # dataset = CorrelationFunction_DESIMockChallenge_Post(
    #    isotropic=False, recon="iso", fit_poles=[0, 2, 4], min_dist=35, max_dist=157.5, num_mocks=998
    # )
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

    model = CorrRoss2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om", "beta", "sigma_s", "sigma_nl_par", "sigma_nl_perp"],
        poly_poles=dataset.fit_poles,
        correction=Correction.HARTLAP,
    )
    model.set_default("beta", 0.4)
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
    p["b{0}_{1}"] = 4.0
    p["b{2}_{1}"] = 4.0
    p["b{4}_{1}"] = 4.0
    print(model.get_alphas(p["alpha"], p["epsilon"]), p)
    sprime, xi, marged = model.compute_correlation_function(data[0]["dist"], p)

    print(data[0]["dist"], xi)
    # np.savetxt("../../barry/data/desi_kp4/test_xi_sigmatest.dat", np.c_[data[0]["dist"], xi[0], xi[1], xi[2]])

    xiells = np.loadtxt("../../barry/data/desi_kp4/xiells.txt").T

    import matplotlib.pyplot as plt

    vfac = 1.0  # / (1.2127500000000002 * 1.0476190476190477**2)
    print(vfac)

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[1])
    plt.errorbar(data[0]["dist"], data[0]["dist"] ** 2 * xi[0] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[2])
    plt.errorbar(data[0]["dist"], data[0]["dist"] ** 2 * xi[1] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()

    plt.plot(xiells[0], xiells[0] ** 2 * xiells[3])
    plt.errorbar(data[0]["dist"], data[0]["dist"] ** 2 * xi[2] * vfac)
    plt.xlim(0.0, 200.0)
    plt.show()"""
