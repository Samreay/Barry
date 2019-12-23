import logging

from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
from scipy import integrate
import numpy as np


class CorrBeutler2017(CorrelationFunctionFit):
    """  xi(s) model inspired from Beutler 2017 and Ross 2017.
    """

    def __init__(self, name="Corr Beutler 2017", recon=False, smooth_type="hinton2017", fix_params=("om", "f"), smooth=False, correction=None, isotropic=True):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, smooth=smooth, correction=correction, isotropic=isotropic)

    def set_data(self, data):
        super().set_data(data)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 1.0, 20.0, 1.0)  # dampening
            self.add_param("a1", r"$a_1$", -100, 100, 0)  # Polynomial marginalisation 1
            self.add_param("a2", r"$a_2$", -2, 2, 0)  # Polynomial marginalisation 2
            self.add_param("a3", r"$a_3$", -0.2, 0.2, 0)  # Polynomial marginalisation 3
        else:
            self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 1.0, 20.0, 1.0)  # dampening parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 1.0, 20.0, 1.0)  # dampening perpendicular to LOS
            self.add_param("a0_1", r"$a0_1$", -100, 100, 0)  # Monopole Polynomial marginalisation 1
            self.add_param("a0_2", r"$a0_2$", -2, 2, 0)  # Monopole Polynomial marginalisation 2
            self.add_param("a0_3", r"$a0_3$", -0.2, 0.2, 0)  # Monopole Polynomial marginalisation 3
            self.add_param("a2_1", r"$a2_1$", -100, 100, 0)  # Quadrupole Polynomial marginalisation 1
            self.add_param("a2_2", r"$a2_2$", -2, 2, 0)  # Quadrupole Polynomial marginalisation 2
            self.add_param("a2_3", r"$a2_3$", -0.2, 0.2, 0)  # Quadrupole Polynomial marginalisation 3

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
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        if self.isotropic:
            fog = 1.0 / (1.0 + ks ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            pk_smooth = pk_smooth_lin * fog

            # Compute the propagator
            C = np.exp(-0.5 * ks ** 2 * p["sigma_nl"] ** 2)
            pk0 = pk_smooth * (1.0 + pk_ratio * C)

            sprime = p["alpha"] * dist
            xi0 = p["b0"] * self.pk2xi_0.__call__(ks, pk0, sprime)
            shape = p["a1"] / (dist ** 2) + p["a2"] / dist + p["a3"]

            xi0 += shape
            xi2 = None

        else:
            # First compute the undilated pk multipoles
            fog = 1.0 / (1.0 + np.outer(self.mu ** 2, ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
            if self.recon:
                kaiser_prefac = 1.0 + np.outer(p["f"] * self.mu ** 2, 1.0 - self.camb.smoothing_kernel)
            else:
                kaiser_prefac = 1.0 + np.outer(p["f"] * self.mu ** 2)
            pk_smooth = kaiser_prefac ** 2 * pk_smooth_lin * fog

            # Compute the propagator
            C = np.exp(np.outer(-0.5 * (self.mu ** 2 * p["sigma_nl_par"] ** 2 + (1.0 - self.mu ** 2) * p["sigma_nl_perp"] ** 2), ks ** 2))
            pk2d = (pk_smooth * (1.0 + pk_ratio) * C).T

            pk0 = integrate.simps(pk2d, self.mu, axis=1)
            pk2 = 3.0 * integrate.simps(pk2d * self.mu ** 2, self.mu, axis=1)
            pk4 = 1.125 * (35.0 * integrate.simps(pk2d * self.mu ** 4, self.mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0)

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
            xi0 = p["b0"] * integrate.simps(xi2d, self.mu, axis=1)
            xi2 = 3.0 * p["b2"] * integrate.simps(xi2d * muprime ** 2, self.mu, axis=1)
            xi2 = 2.5 * (xi2 - xi0)

            shape0 = p["a0_1"] / (dist ** 2) + p["a0_2"] / dist + p["a0_3"]
            shape2 = p["a2_1"] / (dist ** 2) + p["a2_2"] / dist + p["a2_3"]
            xi0 += shape0
            xi2 += shape2

        return sprime, xi0, xi2


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_ROSS_DR12_Z061
    from barry.config import setup_logging

    setup_logging()

    print("Checking isotropic")
    dataset = CorrelationFunction_ROSS_DR12_Z061(isotropic=True)
    model_iso = CorrBeutler2017()
    model_iso.sanity_check(dataset)

    print("Checking anisotropic")
    dataset = CorrelationFunction_ROSS_DR12_Z061(isotropic=False)
    model_aniso = CorrBeutler2017(isotropic=False)
    model_aniso.sanity_check(dataset)
