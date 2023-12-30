import sys

sys.path.append("../..")
import logging
from functools import lru_cache
import numpy as np
from scipy import integrate
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep


class PowerSeo2016(PowerSpectrumFit):
    """P(k) model inspired from Seo 2016.

    See https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.2453S for details.
    """

    def __init__(
        self,
        name="Pk Seo 2016",
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
            marg=marg,
            broadband_type=broadband_type,
            n_data=n_data,
            **kwargs,
        )

        if self.recon_type == "sym" or self.recon_type == "ani":
            raise NotImplementedError("Symmetric and Anisotropic reconstruction not yet available for Seo2016 model")

        self.set_marg(fix_params, poly_poles, self.n_poly, do_bias=False, marg_bias=0)

    def precompute(self, om=None, h0=None, ks=None, pk_lin=None, pk_nonlin_0=None, pk_nonlin_z=None, r_drag=None, s=None):

        if ks is None or pk_lin is None:
            ks = self.camb.ks
            pk_lin = self.camb.get_data()["pk_lin_z"]
        if s is None:
            s = self.camb.smoothing_kernel

        r1, r2 = self.get_Rs(ks)

        # R_1/P_lin, R_2/P_lin
        R1 = ks**2 * integrate.simps(pk_lin * r1, x=ks, axis=1) / (4.0 * np.pi**2)
        R2 = ks**2 * integrate.simps(pk_lin * r2, x=ks, axis=1) / (4.0 * np.pi**2)

        return {
            "sigma": integrate.simps(pk_lin, x=ks) / (6.0 * np.pi**2),
            "sigma_dd": integrate.simps(pk_lin * (1.0 - s) ** 2, x=ks) / (6.0 * np.pi**2),
            "sigma_ss": integrate.simps(pk_lin * s**2, x=ks) / (6.0 * np.pi**2),
            "R1": R1,
            "R2": R2,
        }

    @lru_cache(maxsize=2)
    def get_Rs(self, ks):
        r = np.outer(1.0 / ks, ks)
        R1 = -(1.0 + r**2) / (24.0 * r**2) * (3.0 - 14.0 * r**2 + 3.0 * r**4) + (r**2 - 1.0) ** 4 / (16.0 * r**3) * np.log(
            np.fabs((1.0 + r) / (1.0 - r))
        )
        R2 = (1.0 - r**2) / (24.0 * r**2) * (3.0 - 2.0 * r**2 + 3.0 * r**4) + (r**2 - 1.0) ** 3 * (1.0 + r**2) / (
            16.0 * r**3
        ) * np.log(np.fabs((1.0 + r) / (1.0 - r)))

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

    # @lru_cache(maxsize=4)
    # def get_pt_data(self, om):
    #    return self.PT.get_data(om=om)

    @lru_cache(maxsize=4)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_dd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu**2, self.camb.ks**2) * self.get_pregen("sigma_dd", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_ss(self, om):
        return np.exp(-np.tile(self.camb.ks**2, (self.nmu, 1)) * self.get_pregen("sigma_ss", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_dd", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_dd", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_par(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, self.mu**2) * self.get_pregen("sigma_ss", om) / 2.0)

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_ss", om) / 2.0)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 10.0, 5.0)  # Fingers-of-god damping
        for i in range(self.n_data_poly):
            for pole in self.poly_poles:
                for ip in self.n_poly:
                    self.add_param(f"a{{{pole}}}_{{{ip}}}_{{{i+1}}}", f"$a_{{{pole},{ip},{i+1}}}$", -20000.0, 20000.0, 0)

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None, nopoly=False):
        """Computes the power spectrum model using the Seo et. al., 2016 method

        Parameters
        ----------
        k : np.ndarray
            Array of (undilated) k-values to compute the model at.
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature

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
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation
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

        # We split for isotropic and anisotropic here. They are coded up quite differently to try and make things fast
        if self.isotropic:

            kprime = k if for_corr else k / p["alpha"]

            # Compute the smooth model
            fog = 1.0 / (1.0 + np.outer(self.mu**2, ks**2 * p["sigma_s"] ** 2 / 2.0)) ** 2
            pk_smooth = p["b{0}"] ** 2 * pk_smooth_lin

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            if smooth:
                propagator = np.zeros(len(ks))
            else:
                # Lets round some things for the sake of numerical speed
                om = np.round(p["om"], decimals=5)
                growth = np.round(p["beta"] * p["b{0}"], decimals=5)

                # Compute the BAO damping
                if self.recon:
                    damping_dd = self.get_damping_dd(growth, om)
                    damping_ss = self.get_damping_ss(om)

                    # Compute propagator
                    smooth_prefac = np.tile(self.camb.smoothing_kernel / p["b{0}"], (self.nmu, 1))
                    kaiser_prefac = 1.0 + np.outer(p["beta"] * self.mu**2, 1.0 - self.camb.smoothing_kernel)
                    propagator = (kaiser_prefac * damping_dd + smooth_prefac * (damping_ss - damping_dd)) ** 2
                else:
                    damping = self.get_damping(growth, om)

                    prefac_k = 1.0 + np.tile(
                        3.0 / 7.0 * (self.get_pregen("R1", om) * (1.0 - 4.0 / (9.0 * p["b{0}"])) + self.get_pregen("R2", om)), (self.nmu, 1)
                    )
                    prefac_mu = np.outer(
                        self.mu**2,
                        p["beta"]
                        + 3.0 / 7.0 * growth * self.get_pregen("R1", om) * (2.0 - 1.0 / (3.0 * p["b{0}"]))
                        + 6.0 / 7.0 * growth * self.get_pregen("R2", om),
                    )
                    propagator = ((prefac_k + prefac_mu) * damping) ** 2

            pk1d = integrate.simps(pk_smooth * (fog + pk_ratio * propagator), self.mu, axis=0)
            pk = [splev(kprime, splrep(ks, pk1d))]

        else:
            epsilon = np.round(p["epsilon"], decimals=5)
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.get_muprime(epsilon)
            fog = 1.0 / (1.0 + muprime**2 * kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2

            # Lets round some things for the sake of numerical speed
            om = np.round(p["om"], decimals=5)
            growth = np.round(p["beta"] * p["b{0}"], decimals=5)

            sprime = splev(kprime, splrep(ks, self.camb.smoothing_kernel)) if self.recon else 0.0
            kaiser_prefac = 1.0 + growth / p["b{0}"] * muprime**2 * (1.0 - sprime)

            pk_smooth = p["b{0}"] ** 2 * kaiser_prefac**2 * splev(kprime, splrep(ks, pk_smooth_lin))

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            if smooth:
                pk2d = pk_smooth * fog
            else:
                # Compute the BAO damping
                power_par = 1.0 / (p["alpha"] ** 2 * (1.0 + epsilon) ** 4)
                power_perp = (1.0 + epsilon) ** 2 / p["alpha"] ** 2
                if self.recon:
                    damping_dd = (
                        self.get_damping_aniso_dd_par(growth, om, data_name=data_name) ** power_par
                        * self.get_damping_aniso_dd_perp(om, data_name=data_name) ** power_perp
                    )
                    damping_ss = (
                        self.get_damping_aniso_ss_par(om, data_name=data_name) ** power_par
                        * self.get_damping_aniso_ss_perp(om, data_name=data_name) ** power_perp
                    )

                    # Compute propagator
                    smooth_prefac = sprime / p["b{0}"]
                    propagator = (damping_dd + smooth_prefac / kaiser_prefac * (damping_ss - damping_dd)) ** 2
                else:
                    damping = (
                        self.get_damping_aniso_par(growth, om, data_name=data_name) ** power_par
                        * self.get_damping_aniso_perp(om, data_name=data_name) ** power_perp
                    )

                    R1_kprime = splev(kprime, splrep(ks, self.get_pregen("R1", om)))
                    R2_kprime = splev(kprime, splrep(ks, self.get_pregen("R2", om)))

                    prefac_k = 3.0 / 7.0 * (R1_kprime * (1.0 - 4.0 / (9.0 * p["b{0}"])) + R2_kprime)
                    prefac_mu = muprime**2 * (
                        3.0 / 7.0 * growth * R1_kprime * (2.0 - 1.0 / (3.0 * p["b{0}"])) + 6.0 / 7.0 * growth * R2_kprime
                    )
                    propagator = ((1.0 + prefac_k / kaiser_prefac + prefac_mu / kaiser_prefac) * damping) ** 2

                pk2d = pk_smooth * (fog + splev(kprime, splrep(ks, pk_ratio)) * propagator)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]

        return kprime, pk


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    data = dataset.get_data()

    model = PowerSeo2016(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.HARTLAP,
        n_poly=5,
    )
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.sanity_check(dataset)
