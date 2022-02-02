import logging
import sys

sys.path.append("../..")

from barry.models import PowerDing2018
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
import numpy as np


class CorrDing2018(CorrelationFunctionFit):
    """xi(s) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(
        self,
        name="Corr Ding 2018",
        fix_params=("om", "beta"),
        smooth_type="hinton2017",
        recon=None,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
    ):

        self.recon = False
        self.recon_type = "None"
        if recon is not None:
            if recon.lower() is not "None":
                self.recon_type = "iso"
                if recon.lower() == "ani":
                    self.recon_type = "ani"
                self.recon = True

        self.recon_smoothing_scale = None
        if isotropic:
            poly_poles = [0]
        if marg is not None:
            fix_params = list(fix_params)
            for pole in poly_poles:
                fix_params.extend([f"a{{{pole}}}_1", f"a{{{pole}}}_2", f"a{{{pole}}}_3"])
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
        self.parent = PowerDing2018(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
        )
        if self.marg:
            for pole in self.poly_poles:
                self.set_default(f"a{{{pole}}}_1", 0.0)
                self.set_default(f"a{{{pole}}}_2", 0.0)
                self.set_default(f"a{{{pole}}}_3", 0.0)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("b", r"$b$", 0.1, 10.0, 1.0)  # Galaxy bias
        self.add_param("beta", r"$\beta$", 0.01, 4.0, 0.5)  # RSD parameter f/b
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", -5.0, 5.0, 0.0)  # Non-linear galaxy bias
        for pole in self.poly_poles:
            self.add_param(f"a{{{pole}}}_1", f"$a_{{{pole},1}}$", -100.0, 100.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param(f"a{{{pole}}}_2", f"$a_{{{pole},2}}$", -2.0, 2.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param(f"a{{{pole}}}_3", f"$a_{{{pole},3}}$", -0.2, 0.2, 0)  # Monopole Polynomial marginalisation 3

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the correlation function model using the Ding et. al., 2018 EFT0 model power spectrum
            and 3 polynomial terms per multipole

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
        xi, poly = self.add_three_poly(dist, p, xi_comp)

        return sprime, xi, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import CorrelationFunction_ROSS_DR12
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    print("Checking isotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=True, recon="iso", realisation="data")
    model = CorrDing2018(recon=dataset.recon, marg="full", isotropic=dataset.isotropic, correction=Correction.NONE)
    model.sanity_check(dataset)

    print("Checking anisotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2], realisation="data")
    model = CorrDing2018(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        poly_poles=[0, 2],
        correction=Correction.NONE,
    )
    model.sanity_check(dataset)
