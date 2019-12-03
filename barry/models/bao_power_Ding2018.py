import logging
from functools import lru_cache
import numpy as np
from scipy import integrate
from scipy.special import jn
from barry.models.bao_power import PowerSpectrumFit


class PowerDing2018(PowerSpectrumFit):
    """ P(k) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(self, name="Pk Ding 2018", fix_params=("om", "f"), smooth_type="hinton2017", recon=False, postprocess=None, smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

    def precompute(self, camb, om, h0):

        c = camb.get_data(om, h0)
        r_drag = c["r_s"]
        ks = c["ks"]
        pk_lin = c["pk_lin"]
        j0 = jn(0, r_drag * ks)
        s = camb.smoothing_kernel

        return {
            "sigma_nl": integrate.simps(pk_lin * (1.0 - j0), ks) / (6.0 * np.pi ** 2),
            "sigma_dd_nl": integrate.simps(pk_lin * (1.0 - s) ** 2 * (1.0 - j0), ks) / (6.0 * np.pi ** 2),
            "sigma_sd_nl": integrate.simps(pk_lin * (0.5 * (s ** 2 + (1.0 - s) ** 2) - j0 * s * (1.0 - s)), ks) / (6.0 * np.pi ** 2),
            "sigma_ss_nl": integrate.simps(pk_lin * s ** 2 * (1.0 - j0), ks) / (6.0 * np.pi ** 2),
        }

    @lru_cache(maxsize=32)
    def get_damping_dd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_dd_nl", om))

    @lru_cache(maxsize=32)
    def get_damping_sd(self, growth, om):
        return np.exp(-np.outer(1.0 + growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_sd_nl", om))

    @lru_cache(maxsize=32)
    def get_damping_ss(self, om):
        return np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.get_pregen("sigma_ss_nl", om))

    @lru_cache(maxsize=32)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_nl", om))

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0, 5.0)  # Non-linear galaxy bias
        self.add_param("a1", r"$a_1$", -20000.0, 15000.0, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -15000.0, 25000.0, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -5000.0, 5000.0, 0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -200.0, 200.0, 0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -3.0, 3.0, 0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, p, smooth=False, shape=True):
        """ Computes the power spectrum model using the Ding et. al., 2018 EFT0 model
        
        Parameters
        ----------
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether or not to return a smooth pk without BAO feature
        shape : bool, optional
            Whether or not to add in shape terms

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
            shape = 0  # Its vectorised, don't worry

        if smooth:
            pk1d = integrate.simps((pk_smooth + shape), self.mu, axis=0)
        else:
            # Lets round some things for the sake of numerical speed
            om = np.round(p["om"], decimals=5)
            growth = np.round(growth, decimals=5)

            # Compute the BAO damping
            if self.recon:
                damping_dd = self.get_damping_dd(growth, om)
                damping_sd = self.get_damping_sd(growth, om)
                damping_ss = self.get_damping_ss(om)

                smooth_prefac = np.tile(self.camb.smoothing_kernel / p["b"], (self.nmu, 1))
                bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
                kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth / p["b"] * self.mu ** 2, 1.0 - self.camb.smoothing_kernel) + bdelta_prefac
                propagator = (
                    (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping_dd + 2.0 * kaiser_prefac * smooth_prefac * damping_sd + smooth_prefac ** 2 * damping_ss
                )
            else:
                damping = self.get_damping(growth, om)
                bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
                kaiser_prefac = 1.0 + np.tile(growth / p["b"] * self.mu ** 2, (len(ks), 1)).T + bdelta_prefac
                propagator = (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping

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
    model_pre = PowerDing2018(recon=False)
    model_pre.sanity_check(dataset)

    print("Checking post-recon")
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True)
    model_post = PowerDing2018(recon=True)
    model_post.sanity_check(dataset)
