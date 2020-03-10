import logging
from functools import lru_cache
import numpy as np
from scipy import integrate
from scipy.special import jn
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep


class PowerDing2018(PowerSpectrumFit):
    """ P(k) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(
        self,
        name="Pk Ding 2018",
        fix_params=("om", "f"),
        smooth_type="hinton2017",
        recon=False,
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=True,
    ):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(
            name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction, isotropic=isotropic
        )

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

    @lru_cache(maxsize=4)
    def get_damping_dd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_dd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -(1.0 + (2.0 + growth) * growth) * np.outer(ks ** 2, self.mu ** 2) * self.get_pregen("sigma_dd_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -np.outer(ks ** 2, 1.0 - self.mu ** 2) * self.get_pregen("sigma_dd_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_sd(self, growth, om):
        return np.exp(-np.outer(1.0 + growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_sd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -(1.0 + growth) * np.outer(ks ** 2, self.mu ** 2) * self.get_pregen("sigma_sd_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -np.outer(ks ** 2, 1.0 - self.mu ** 2) * self.get_pregen("sigma_sd_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_ss(self, om):
        return np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.get_pregen("sigma_ss_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_par(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -np.outer(ks ** 2, self.mu ** 2) * self.get_pregen("sigma_ss_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -np.outer(ks ** 2, 1.0 - self.mu ** 2) * self.get_pregen("sigma_dd_nl", om)

    @lru_cache(maxsize=4)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pregen("sigma_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -(1.0 + (2.0 + growth) * growth) * np.outer(ks ** 2, self.mu ** 2) * self.get_pregen("sigma_nl", om)

    @lru_cache(maxsize=4)
    def get_damping_aniso_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return -np.outer(ks ** 2, 1.0 - self.mu ** 2) * self.get_pregen("sigma_nl", om)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0, 5.0)  # Non-linear galaxy bias
        if self.isotropic:
            self.add_param("a1", r"$a_1$", -10000.0, 30000.0, 0)  # Polynomial marginalisation 1
            self.add_param("a2", r"$a_2$", -20000.0, 10000.0, 0)  # Polynomial marginalisation 2
            self.add_param("a3", r"$a_3$", -1000.0, 5000.0, 0)  # Polynomial marginalisation 3
            self.add_param("a4", r"$a_4$", -200.0, 200.0, 0)  # Polynomial marginalisation 4
            self.add_param("a5", r"$a_5$", -3.0, 3.0, 0)  # Polynomial marginalisation 5
        else:
            self.add_param("a0_1", r"$a_{0,1}$", -10000.0, 30000.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param("a0_2", r"$a_{0,2}$", -20000.0, 10000.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param("a0_3", r"$a_{0,3}$", -1000.0, 5000.0, 0)  # Monopole Polynomial marginalisation 3
            self.add_param("a0_4", r"$a_{0,4}$", -200.0, 200.0, 0)  # Monopole Polynomial marginalisation 4
            self.add_param("a0_5", r"$a_{0,5}$", -3.0, 3.0, 0)  # Monopole Polynomial marginalisation 5
            self.add_param("a2_1", r"$a_{2,1}$", -10000.0, 30000.0, 0)  # Quadrupole Polynomial marginalisation 1
            self.add_param("a2_2", r"$a_{2,2}$", -20000.0, 10000.0, 0)  # Quadrupole Polynomial marginalisation 2
            self.add_param("a2_3", r"$a_{2,3}$", -1000.0, 5000.0, 0)  # Quadrupole Polynomial marginalisation 3
            self.add_param("a2_4", r"$a_{2,4}$", -200.0, 200.0, 0)  # Quadrupole Polynomial marginalisation 4
            self.add_param("a2_5", r"$a_{2,5}$", -3.0, 3.0, 0)  # Quadrupole Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p, smooth=False, dilate=True, data_name=None):
        """ Computes the power spectrum model using the Ding et. al., 2018 EFT0 propagator

        Parameters
        ----------
        k : np.ndarray
            Array of (undilated) k-values to compute the model at.
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature
        shape : bool, optional
            Whether or not to include shape marginalisation terms.
        dilate : bool, optional
            Whether or not to dilate the k-values of the model based on the values of alpha (and epsilon)

        Returns
        -------
        kprime : np.ndarray
            Wavenumbers of the computed pk
        pk0 : np.ndarray
            the model monopole interpolated to kprime.
        pk2 : np.ndarray
            the model quadrupole interpolated to kprime. Will be 'None' if the model is isotropic
        pk4 : np.ndarray
            the model hexadecapole interpolated to kprime. Will be 'None' if the model is isotropic

        """

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # Compute the growth rate depending on what we have left as free parameters
        growth = p["f"]

        if self.isotropic:
            # Compute the smooth model
            fog = 1.0 / (1.0 + np.outer(self.mu ** 2, ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
            pk_smooth = p["b"] ** 2 * pk_smooth_lin * fog

            # Polynomial shape
            if self.recon:
                shape = p["a1"] * ks ** 2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
            else:
                shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

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

                    smooth_prefac = np.tile(s / p["b"], (self.nmu, 1))
                    bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
                    kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth / p["b"] * self.mu ** 2, 1.0 - s) + bdelta_prefac
                    propagator = (
                        (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping_dd
                        + 2.0 * kaiser_prefac * smooth_prefac * damping_sd
                        + smooth_prefac ** 2 * damping_ss
                    )
                else:
                    damping = self.get_damping(growth, om)
                    bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
                    kaiser_prefac = 1.0 + np.tile(growth / p["b"] * self.mu ** 2, (len(ks), 1)).T + bdelta_prefac
                    propagator = (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping

                pk1d = integrate.simps((pk_smooth + shape) * (1.0 + pk_ratio * propagator), self.mu, axis=0)

            if dilate:
                kprime = k / p["alpha"]
            else:
                kprime = k

            pk0 = splev(kprime, splrep(ks, pk1d))
            pk2 = None
            pk4 = None

        else:
            epsilon = np.round(p["epsilon"], decimals=5)
            kprime = np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.get_muprime(epsilon)
            fog = 1.0 / (1.0 + muprime ** 2 * kprime ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            pk_smooth = p["b"] ** 2 * splev(kprime, splrep(ks, pk_smooth_lin)) * fog

            if smooth:
                pk2d = pk_smooth
            else:
                # Lets round some things for the sake of numerical speed
                om = np.round(p["om"], decimals=5)
                growth = np.round(p["f"], decimals=5)

                # Compute the BAO damping
                if self.recon:
                    prop_prefac = np.exp(1.0 / (p["alpha"] ** 2 * (1.0 + epsilon) ** 4))
                    damping_dd = np.exp(
                        self.get_damping_aniso_dd_par(growth, om, data_name=data_name)
                        + self.get_damping_aniso_dd_perp(om, data_name=data_name) * (1.0 + epsilon) ** 6
                    )
                    damping_sd = np.exp(
                        self.get_damping_aniso_sd_par(growth, om, data_name=data_name)
                        + self.get_damping_aniso_sd_perp(om, data_name=data_name) * (1.0 + epsilon) ** 6
                    )
                    damping_ss = np.exp(
                        self.get_damping_aniso_ss_par(om, data_name=data_name) + self.get_damping_aniso_ss_perp(om, data_name=data_name) * (1.0 + epsilon) ** 6
                    )
                    sprime = splev(kprime, splrep(ks, self.camb.smoothing_kernel))

                    # Compute propagator
                    smooth_prefac = sprime / p["b"]
                    bdelta_prefac = 0.5 * p["b_delta"] / p["b"] * kprime ** 2
                    kaiser_prefac = 1.0 - smooth_prefac + growth / p["b"] * muprime ** 2 * (1.0 - sprime) + bdelta_prefac
                    propagator = prop_prefac * (
                        (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping_dd
                        + 2.0 * kaiser_prefac * smooth_prefac * damping_sd
                        + smooth_prefac ** 2 * damping_ss
                    )
                else:
                    prop_prefac = np.exp(1.0 / (p["alpha"] ** 2 * (1.0 + epsilon) ** 4))
                    damping = np.exp(
                        self.get_damping_aniso_par(growth, om, data_name=data_name)
                        + self.get_damping_aniso_perp(om, data_name=data_name) * (1.0 + epsilon) ** 6
                    )

                    # Compute propagator
                    bdelta_prefac = 0.5 * p["b_delta"] / p["b"] * kprime ** 2
                    kaiser_prefac = 1.0 + growth / p["b"] * muprime ** 2 + bdelta_prefac
                    propagator = prop_prefac * ((kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping)

                pk2d = pk_smooth * (1.0 + splev(kprime, splrep(ks, pk_ratio)) * propagator)

            # Polynomial shape
            if self.recon:
                shape0 = p["a0_1"] * k ** 2 + p["a0_2"] + p["a0_3"] / k + p["a0_4"] / (k * k) + p["a0_5"] / (k ** 3)
                shape2 = p["a2_1"] * k ** 2 + p["a2_2"] + p["a2_3"] / k + p["a2_4"] / (k * k) + p["a2_5"] / (k ** 3)
            else:
                shape0 = p["a0_1"] * k + p["a0_2"] + p["a0_3"] / k + p["a0_4"] / (k * k) + p["a0_5"] / (k ** 3)
                shape2 = p["a2_1"] * k + p["a2_2"] + p["a2_3"] / k + p["a2_4"] / (k * k) + p["a2_5"] / (k ** 3)

            pk0 = integrate.simps(pk2d, self.mu, axis=1)
            pk2 = 3.0 * integrate.simps(pk2d * self.mu ** 2, self.mu, axis=1)
            pk4 = 1.125 * (35.0 * integrate.simps(pk2d * self.mu ** 4, self.mu, axis=1) - 10.0 * pk2 + 3.0 * pk0)
            pk2 = 2.5 * (pk2 - pk0) + shape2
            pk0 = pk0 + shape0

        return kprime, pk0, pk2, pk4


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_Beutler2019_Z061_SGC
    from barry.config import setup_logging

    setup_logging()

    print("Getting default 1D")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=True)
    model_post = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_post.plot_default(dataset)

    print("Getting default 2D")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=False)
    model_post = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_post.plot_default(dataset)

    """print("Checking isotropic mock mean")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=True)
    model_pre = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_pre.sanity_check(dataset)

    print("Checking isotropic data")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=True, realisation="data")
    model_post = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_post.sanity_check(dataset)"""

    print("Checking anisotropic mock mean")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=False)
    model_post = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_post.sanity_check(dataset)

    print("Checking anisotropic data")
    dataset = PowerSpectrum_Beutler2019_Z061_SGC(isotropic=False, realisation="data")
    model_post = PowerDing2018(recon=dataset.recon, isotropic=dataset.isotropic)
    model_post.sanity_check(dataset)
