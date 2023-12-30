import sys

sys.path.append("../..")
import logging
from functools import lru_cache
import numpy as np
from scipy import integrate
from scipy.special import spherical_jn
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt


class PowerChen2019(PowerSpectrumFit):
    """P(k) model inspired from Chen 2019.

    See https://ui.adsabs.harvard.edu/abs/2019JCAP...09..017C/abstract for details.

    """

    def __init__(
        self,
        name="Pk Chen 2019",
        fix_params=("om", "beta"),
        smooth_type=None,
        recon=None,
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        broadband_type="spline",
        n_data=1,
        **kwargs,
    ):

        self.marg_bias = 0

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            postprocess=postprocess,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            broadband_type=broadband_type,
            n_data=n_data,
            **kwargs,
        )

        if self.recon_type == "ani":
            raise NotImplementedError("Anisotropic reconstruction not yet available for Chen2019 model")

        self.set_marg(fix_params, poly_poles, self.n_poly, do_bias=False, marg_bias=0)

    def precompute(self, om=None, h0=None, ks=None, pk_lin=None, pk_nonlin_0=None, pk_nonlin_z=None, r_drag=None, s=None):

        if ks is None or pk_lin is None:
            ks = self.camb.ks
            pk_lin = self.camb.get_data()["pk_lin_z"]
        if r_drag is None:
            r_drag = self.camb.get_data()["r_s"]
        if s is None:
            s = self.camb.smoothing_kernel

        j0 = spherical_jn(0, r_drag * ks)

        return {
            "sigma_nl": integrate.simps(pk_lin * (1.0 - j0), ks) / (6.0 * np.pi**2),
            "sigma_dd_nl": integrate.simps(pk_lin * (1.0 - s) ** 2 * (1.0 - j0), ks) / (6.0 * np.pi**2),
            "sigma_sd_nl": integrate.simps(pk_lin * (0.5 * (s**2 + (1.0 - s) ** 2) + j0 * s * (1.0 - s)), ks) / (6.0 * np.pi**2),
            "sigma_ss_nl": integrate.simps(pk_lin * s**2 * (1.0 - j0), ks) / (6.0 * np.pi**2),
            "sigma_sd_dd": integrate.simps(pk_lin * 0.5 * (1.0 - s) ** 2, ks) / (6.0 * np.pi**2),
            "sigma_sd_sd": integrate.simps(pk_lin * j0 * s * (1.0 - s), ks) / (6.0 * np.pi**2),
            "sigma_sd_ss": integrate.simps(pk_lin * 0.5 * s**2, ks) / (6.0 * np.pi**2),
        }

    @lru_cache(maxsize=4)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_dd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma_dd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_sd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma_sd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_sd_iso(self, growth, om):
        inner = (
            (1.0 + (2.0 + growth) * growth * self.mu**2) * self.get_pregen("sigma_sd_dd", om)
            + (1.0 + growth * self.mu**2) * self.get_pregen("sigma_sd_sd", om)
            + self.get_pregen("sigma_sd_ss", om)
        )
        return np.exp(-np.outer(inner, self.camb.ks**2))

    @lru_cache(maxsize=4)
    def get_damping_ss(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma_ss_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_dd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_dd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_sd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_sd_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_dd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_sd_dd", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_dd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_sd_dd", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_sd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + growth) * ks**2, self.mu**2) * self.get_pregen("sigma_sd_sd", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_sd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_sd_sd", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_ss_par(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, self.mu**2) * self.get_pregen("sigma_sd_ss", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_sd_ss_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_sd_ss", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_ss_nl", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks if self.kvals is None else self.kvals
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_ss_nl", om))

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("beta", r"$\beta$", 0.01, 1.0, None)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 10.0, 5.0)  # Fingers-of-god damping
        for i in range(self.n_data_poly):
            for pole in self.poly_poles:
                for ip in self.n_poly:
                    self.add_param(f"a{{{pole}}}_{{{ip}}}_{{{i+1}}}", f"$a_{{{pole},{ip},{i+1}}}$", -20000.0, 20000.0, 0)

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None, nopoly=False):
        """Computes the power spectrum model using the Chen et. al., 2019 propagator

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
        if self.kvals is None or self.pksmooth is None or self.pkratio is None:
            ks = self.camb.ks
            pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])
        else:
            ks = self.kvals
            pk_smooth_lin, pk_ratio = self.pksmooth, self.pkratio

        if not for_corr:
            if "b{0}" not in p:
                p = self.deal_with_ndata(p, 0)

        if self.isotropic:

            kprime = k if for_corr else k / p["alpha"]

            # Compute the smooth model
            fog = 1.0 / (1.0 + np.outer(self.mu**2, ks**2 * p["sigma_s"] ** 2 / 2.0)) ** 2
            pk_smooth = p["b{0}"] * pk_smooth_lin

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            if smooth:
                propagator = np.zeros(len(ks))
            else:
                # Lets round some things for the sake of numerical speed
                om = np.round(p["om"], decimals=5)
                growth = np.round(p["b{0}"] * p["beta"], decimals=5)

                # Compute the BAO damping
                if self.recon:
                    damping_dd = self.get_damping_dd(growth, om)
                    damping_sd = self.get_damping_sd_iso(growth, om) if self.recon_type == "iso" else self.get_damping_sd(growth, om)
                    damping_ss = self.get_damping_ss(growth, om)

                    dd_prefac = (
                        1.0 + np.outer(p["beta"] * self.mu**2, 1.0 - self.camb.smoothing_kernel) - self.camb.smoothing_kernel / p["b{0}"]
                    )
                    ss_prefac = (
                        np.outer(1.0 + growth * self.mu**2, self.camb.smoothing_kernel) / p["b{0}"]
                        if self.recon_type == "sym"
                        else self.camb.smoothing_kernel / p["b{0}"]
                    )
                    sd_prefac = dd_prefac * ss_prefac
                    propagator = dd_prefac**2 * damping_dd + 2.0 * sd_prefac * damping_sd + ss_prefac**2 * damping_ss
                else:
                    damping = self.get_damping(growth, om)
                    kaiser_prefac = 1.0 + np.tile(p["beta"] * self.mu**2, (len(ks), 1)).T
                    propagator = kaiser_prefac**2 * damping

            pk1d = integrate.simps(pk_smooth * (fog + pk_ratio * propagator), self.mu, axis=0)
            pk = [splev(kprime, splrep(ks, pk1d))]

        else:
            epsilon = np.round(p["epsilon"], decimals=5)
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.get_muprime(epsilon)

            # Lets round some things for the sake of numerical speed
            om = np.round(p["om"], decimals=5)
            growth = np.round(p["b{0}"] * p["beta"], decimals=5)

            sprime = splev(kprime, splrep(ks, self.camb.smoothing_kernel)) if self.recon else 0.0

            fog = 1.0 / (1.0 + muprime**2 * kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
            pk_smooth = splev(kprime, splrep(ks, pk_smooth_lin))

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            dd_prefac = p["b{0}"] + growth * muprime**2 * (1.0 - sprime) - sprime
            ss_prefac = 0.0 if not self.recon else (1.0 + growth * muprime**2) * sprime if self.recon_type == "sym" else sprime
            broadband = (dd_prefac + ss_prefac) ** 2 * fog

            if smooth:
                pk2d = pk_smooth * broadband
            else:
                # Compute the BAO damping
                power_par = 1.0 / (p["alpha"] ** 2 * (1.0 + epsilon) ** 4)
                power_perp = (1.0 + epsilon) ** 2 / p["alpha"] ** 2
                if self.recon:
                    damping_dd = (
                        self.get_damping_aniso_dd_par(growth, om, data_name=data_name) ** power_par
                        * self.get_damping_aniso_dd_perp(om, data_name=data_name) ** power_perp
                    )
                    if self.recon_type == "iso":
                        damping_sd = (
                            self.get_damping_aniso_sd_dd_par(growth, om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_sd_dd_perp(om, data_name=data_name) ** power_perp
                            * self.get_damping_aniso_sd_sd_par(growth, om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_sd_sd_perp(om, data_name=data_name) ** power_perp
                            * self.get_damping_aniso_sd_ss_par(om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_sd_ss_perp(om, data_name=data_name) ** power_perp
                        )
                        damping_ss = (
                            self.get_damping_aniso_ss_par(0.0, om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_ss_perp(om, data_name=data_name) ** power_perp
                        )
                    else:
                        damping_sd = (
                            self.get_damping_aniso_sd_par(growth, om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_sd_perp(om, data_name=data_name) ** power_perp
                        )
                        damping_ss = (
                            self.get_damping_aniso_ss_par(growth, om, data_name=data_name) ** power_par
                            * self.get_damping_aniso_ss_perp(om, data_name=data_name) ** power_perp
                        )

                    # Compute propagator
                    propagator = dd_prefac**2 * damping_dd + 2.0 * dd_prefac * ss_prefac * damping_sd + ss_prefac**2 * damping_ss
                else:
                    dd_prefac = p["b{0}"] + growth * muprime**2
                    damping = (
                        self.get_damping_aniso_par(growth, om, data_name=data_name) ** power_par
                        * self.get_damping_aniso_perp(om, data_name=data_name) ** power_perp
                    )

                    # Compute propagator
                    propagator = dd_prefac**2 * damping
                    broadband = dd_prefac**2 * fog

                pk2d = pk_smooth * (broadband + splev(kprime, splrep(ks, pk_ratio)) * propagator)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]

        return kprime, pk


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12, PowerSpectrum_DESI_KP4
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    print("Checking anisotropic mock mean")
    dataset = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    model = PowerChen2019(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        broadband_type="spline",
    )
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.sanity_check(dataset)
