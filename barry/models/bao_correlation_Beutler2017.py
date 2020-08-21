import logging

from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
from scipy import integrate
import numpy as np


class CorrBeutler2017(CorrelationFunctionFit):
    """  xi(s) model inspired from Beutler 2017 and Ross 2017.
    """

    def __init__(
        self,
        name="Corr Beutler 2017",
        fix_params=("om"),
        smooth_type="hinton2017",
        recon=False,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
    ):
        self.recon = recon
        self.recon_smoothing_scale = None
        if isotropic:
            poly_poles = [0]
        if marg is not None:
            fix_params = list(fix_params)
            for pole in poly_poles:
                fix_params.extend([f"b{{{pole}}}"])
                fix_params.extend([f"a{{{pole}}}_1", f"a{{{pole}}}_2", f"a{{{pole}}}_3", f"a{{{pole}}}_4", f"a{{{pole}}}_5"])
        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
        )
        self.parent = PowerBeutler2017(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
        )
        if self.marg:
            for pole in self.poly_poles:
                self.set_default(f"b{{{pole}}}", 1.0)
                self.set_default(f"a{{{pole}}}_1", 0.0)
                self.set_default(f"a{{{pole}}}_2", 0.0)
                self.set_default(f"a{{{pole}}}_3", 0.0)

    def set_data(self, data):
        super().set_data(data)
        self.parent.set_data(data)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.01, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 1.0, 0.5)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for pole in self.poly_poles:
            self.add_param(f"a{{{pole}}}_1", f"$a_{{{pole},1}}$", -100.0, 100.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param(f"a{{{pole}}}_2", f"$a_{{{pole},2}}$", -2.0, 2.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param(f"a{{{pole}}}_3", f"$a_{{{pole},3}}$", -0.2, 0.2, 0)  # Monopole Polynomial marginalisation 3

    def compute_correlation_function(self, dist, p, smooth=False):
        """ Computes the correlation function model using the Beutler et. al., 2017 power spectrum and Ross 2017 method

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
        xi0 : np.ndarray
            the model monopole interpolated to sprime.
        xi2 : np.ndarray
            the model quadrupole interpolated to sprime. Will be 'None' if the model is isotropic

        """
        ks, pk0, pk2, pk4, _ = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)
        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        if self.isotropic:
            sprime = p["alpha"] * dist
            xi0 = p["b{0}"] * self.pk2xi_0.__call__(ks, pk0, sprime)
            xi[0] = xi0 + p["a{0}_1"] / (dist ** 2) + p["a{0}_2"] / dist + p["a{0}_3"]

            poly = np.zeros((1, len(dist)))
            if self.marg:
                poly = [xi0, 1.0 / (dist ** 2), 1.0 / dist, np.ones(len(dist))]

        else:
            # Construct the dilated 2D correlation function by splining the undilated multipoles. We could have computed these
            # directly at sprime, but sprime depends on both s and mu, so splining is quicker
            epsilon = np.round(p["epsilon"], decimals=5)
            sprime = np.outer(dist * p["alpha"], self.get_sprimefac(epsilon))
            muprime = self.get_muprime(epsilon)

            xi0 = splev(sprime, splrep(dist, self.pk2xi_0.__call__(ks, pk0, dist)))
            xi2 = splev(sprime, splrep(dist, self.pk2xi_2.__call__(ks, pk2, dist)))
            xi4 = splev(sprime, splrep(dist, self.pk2xi_4.__call__(ks, pk4, dist)))

            xi2d = xi0 + 0.5 * (3.0 * muprime ** 2 - 1) * xi2 + 0.125 * (35.0 * muprime ** 4 - 30.0 * muprime ** 2 + 3.0) * xi4

            # Now compute the dilated xi multipoles
            xi0, xi2, xi4 = self.integrate_mu(xi2d)

            xi[0] = p["b{0}"] * xi0
            xi[1] = 2.5 * (p["b{2}"] * xi2 - xi[0])
            if 4 in self.poly_poles:
                xi[2] = 1.125 * (p["b{4}"] * xi4 - 10.0 * p["b{2}"] * xi2 + 3.0 * p["b{0}"] * xi0)
            else:
                xi[2] = 1.125 * (xi4 - 10.0 * p["b{2}"] * xi2 + 3.0 * p["b{0}"] * xi0)

            # Polynomial shape
            if self.marg:
                xi_marg = [xi0, 2.5 * xi2, 1.125 * xi4]
                poly = np.zeros((4 * len(self.poly_poles), 3, len(dist)))
                for pole in self.poly_poles:
                    npole = int(pole / 2)
                    poly[4 * npole : 4 * (npole + 1), npole] = [xi_marg[pole], 1.0 / (dist ** 2), 1.0 / dist, np.ones(len(dist))]
                poly[0, 1] = -2.5 * xi0
                poly[0, 2] = 1.125 * 3.0 * xi0
                if 2 in self.poly_poles:
                    poly[4, 2] = -1.125 * 10.0 * xi2

                xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
            else:
                poly = np.zeros((1, 3, len(dist)))
                for pole in self.poly_poles:
                    xi[int(pole / 2)] += p[f"a{{{pole}}}_1"] / dist ** 2 + p[f"a{{{pole}}}_2"] / dist + p[f"a{{{pole}}}_3"]

        return sprime, xi[0], xi[1], xi[2], poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_ROSS_DR12_Z061
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    """print("Checking isotropic")
    dataset = CorrelationFunction_ROSS_DR12_Z061(isotropic=True, realisation="data")
    model = CorrBeutler2017(recon=dataset.recon, isotropic=dataset.isotropic, fix_params=["om"], correction=Correction.HARTLAP, marg="full")
    model_iso = CorrBeutler2017(recon=dataset.recon, isotropic=dataset.isotropic, fix_params=["om"], correction=Correction.HARTLAP)
    model.sanity_check(dataset)
    model_iso.sanity_check(dataset)"""

    print("Checking anisotropic")
    dataset = CorrelationFunction_ROSS_DR12_Z061(isotropic=False, min_dist=50, max_dist=150, realisation="data")
    model = CorrBeutler2017(recon=dataset.recon, isotropic=dataset.isotropic, fix_params=["om"], correction=Correction.HARTLAP, marg="full")
    model_aniso = CorrBeutler2017(recon=dataset.recon, isotropic=dataset.isotropic, fix_params=["om"], correction=Correction.HARTLAP)
    model.sanity_check(dataset)
    model_aniso.sanity_check(dataset)
