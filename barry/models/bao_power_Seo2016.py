import logging
from functools import lru_cache

import numpy as np
from scipy import integrate
from barry.models.bao_power import PowerSpectrumFit


class PowerSeo2016(PowerSpectrumFit):
    """ P(k) model inspired from Seo 2016.

    See https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.2453S for details.
    """

    def __init__(self, name="Pk Seo 2016", fix_params=("om", "f"), smooth_type="hinton2017", recon=False, postprocess=None, smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

    def precompute(self, camb, om, h0):

        c = camb.get_data(om, h0)
        ks = c["ks"]
        pk_lin = c["pk_lin"]
        s = camb.smoothing_kernel

        r1, r2 = self.get_Rs()

        # R_1/P_lin, R_2/P_lin
        R1 = ks ** 2 * integrate.simps(pk_lin * r1, ks, axis=1) / (4.0 * np.pi ** 2)
        R2 = ks ** 2 * integrate.simps(pk_lin * r2, ks, axis=1) / (4.0 * np.pi ** 2)

        return {
            "sigma": integrate.simps(pk_lin, ks) / (6.0 * np.pi ** 2),
            "sigma_dd": integrate.simps(pk_lin * (1.0 - s) ** 2, ks) / (6.0 * np.pi ** 2),
            "sigma_ss": integrate.simps(pk_lin * s ** 2, ks) / (6.0 * np.pi ** 2),
            "R1": R1,
            "R2": R2,
        }

    @lru_cache(maxsize=2)
    def get_Rs(self):
        ks = self.camb.ks
        r = np.outer(1.0 / ks, ks)
        R1 = -(1.0 + r ** 2) / (24.0 * r ** 2) * (3.0 - 14.0 * r ** 2 + 3.0 * r ** 4) + (r ** 2 - 1.0) ** 4 / (16.0 * r ** 3) * np.log(
            np.fabs((1.0 + r) / (1.0 - r))
        )
        R2 = (1.0 - r ** 2) / (24.0 * r ** 2) * (3.0 - 2.0 * r ** 2 + 3.0 * r ** 4) + (r ** 2 - 1.0) ** 3 * (1.0 + r ** 2) / (16.0 * r ** 3) * np.log(
            np.fabs((1.0 + r) / (1.0 - r))
        )

        # We get NaNs in R1, R2 etc., when r = 1.0 (diagonals). We manually set these to the correct values.
        # We also get numerical issues for large/small r, so we set these manually to asymptotic limits
        R1[np.diag_indices(len(ks))] = 2.0 / 3.0
        R2[np.diag_indices(len(ks))] = 0.0
        index = np.where(r < 1.0e-3)
        R1[index] = 16.0 / 15.0 * r[index] ** 2
        R2[index] = 4.0 / 15.0 * r[index] ** 2
        index = np.where(r > 1.0e2)
        R1[index] = 16.0 / 15.0
        R2[index] = 4.0 / 15.0
        return R1, R2

    @lru_cache(maxsize=32)
    def get_pt_data(self, om):
        return self.PT.get_data(om=om)

    @lru_cache(maxsize=32)
    def get_damping_dd(self, growth, om):
        print(self.nmu, self.mu.shape, self.camb.ks.shape)
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_dd", om) / 2.0)

    @lru_cache(maxsize=32)
    def get_damping_ss(self, om):
        return np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.get_pregen("sigma_ss", om) / 2.0)

    @lru_cache(maxsize=32)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma", om) / 2.0)

    def set_data(self, data):
        super().set_data(data)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("a1", r"$a_1$", -20000.0, 15000.0, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -15000.0, 25000.0, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -5000.0, 5000.0, 0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -200.0, 200.0, 0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -3.0, 3.0, 0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, p, smooth=False, shape=True):
        """ Computes the power spectrum model using the LPT based propagators from Seo et. al., 2016 at k/alpha
        
        Parameters
        ----------
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature
        shape : bool, optional
            Whether or not to include shape marginalisation terms.


        Returns
        -------
        ks : np.ndarray
            Wavenumbers of the computed pk
        pk_1d : np.ndarray
            the ratio (pk_lin / pk_smooth - 1.0),  NOT interpolated to k/alpha.

        """

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # Compute the growth rate depending on what we have left as free parameters
        growth = p["f"]

        # Compute the smooth model
        fog = 1.0 / (1.0 + np.outer(self.mu ** 2, ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
        pk_smooth = p["b"] ** 2 * pk_smooth_lin * fog

        # Polynomial shape
        if shape:
            if self.recon:
                shape = p["a1"] * ks ** 2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
            else:
                shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = 0

        if smooth:
            pk1d = integrate.simps((pk_smooth + shape), self.mu, axis=0)
        else:
            # Lets round some things for the sake of numerical speed
            om = np.round(p["om"], decimals=5)
            growth = np.round(growth, decimals=5)

            # Compute the BAO damping
            if self.recon:
                damping_dd = self.get_damping_dd(growth, om)
                damping_ss = self.get_damping_ss(om)
                s = self.camb.smoothing_kernel

                # Compute propagator
                smooth_prefac = np.tile(s / p["b"], (self.nmu, 1))
                kaiser_prefac = 1.0 + np.outer(growth / p["b"] * self.mu ** 2, 1.0 - s)
                propagator = (kaiser_prefac * damping_dd + smooth_prefac * (damping_ss - damping_dd)) ** 2
            else:
                damping = self.get_damping(growth, om)

                prefac_k = 1.0 + np.tile(3.0 / 7.0 * (self.get_pregen("R1", om) * (1.0 - 4.0 / (9.0 * p["b"])) + self.get_pregen("R2", om)), (self.nmu, 1))
                prefac_mu = np.outer(
                    self.mu ** 2,
                    growth / p["b"]
                    + 3.0 / 7.0 * growth * self.get_pregen("R1", om) * (2.0 - 1.0 / (3.0 * p["b"]))
                    + 6.0 / 7.0 * growth * self.get_pregen("R2", om),
                )
                propagator = ((prefac_k + prefac_mu) * damping) ** 2
            pk1d = integrate.simps((pk_smooth + shape) * (1.0 + pk_ratio * propagator), self.mu, axis=0)
        return ks, pk1d


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12_Z061_NGC
    from barry.config import setup_logging

    setup_logging()

    print("Checking pre-recon")
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=False)
    model_pre = PowerSeo2016(recon=False)
    model_pre.sanity_check(dataset)

    print("Checking post-recon")
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True)
    model_post = PowerSeo2016(recon=True)
    model_post.sanity_check(dataset)
