import logging
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.interpolate import splrep, splev
from scipy.special import jn

from barry.cosmology.power_spectrum_smoothing import smooth_func
from barry.models.bao_power import PowerSpectrumFit
from barry.cosmology.camb_generator import Omega_m_z


class PowerNoda2019(PowerSpectrumFit):
    """P(k) model inspired from Noda 2019.

    See https://ui.adsabs.harvard.edu/abs/2019arXiv190106854N for details.

    """

    def __init__(
        self,
        name="Pk Noda 2019",
        fix_params=("om", "f", "gamma"),
        gammaval=None,
        smooth_type="hinton2017",
        nonlinear_type="spt",
        recon=None,
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=True,
        n_data=1,
    ):
        self.recon = recon
        if gammaval is None:
            if self.recon:
                gammaval = 4.0
            else:
                gammaval = 1.0

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            postprocess=postprocess,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=None,
            n_poly=0,
            n_data=n_data,
            data_share_poly=False,
        )

        if self.recon_type == "ani":
            raise NotImplementedError("Anisotropic reconstruction not yet available for Noda2019 model")

        self.set_default("gamma", gammaval)

        self.nonlinear_type = nonlinear_type.lower()
        if not self.validate_nonlinear_method():
            exit(0)

        self.set_marg(fix_params, [], 0, do_bias=True)

    def get_unique_cosmo_name(self):
        return self.__class__.__name__ + "_" + self.camb.filename_unique + "_" + self.smooth_type + ".pkl"

    def precompute(self, om=None, h0=None, ks=None, pk_lin=None, pk_nonlin_0=None, pk_nonlin_z=None, r_drag=None, s=None):

        if om is None or h0 is None or ks is None or pk_lin is None or pk_nonlin_0 is None or pk_nonlin_z is None:
            c = self.camb.get_data()
            om = c["om"]
            h0 = c["h0"]
            ks = c["ks"]
            pk_lin = c["pk_lin"]
            pk_nonlin_0 = c["pk_nl_0"]
            pk_nonlin_z = c["pk_nl_z"]
        if r_drag is None:
            r_drag = self.camb.get_data()["r_s"]

        r, xs, J00, J01, J11 = self.get_extra()

        # Get the spherical bessel functions
        j0 = jn(0, r_drag * ks)
        j2 = jn(2, r_drag * ks)

        # Get the smoothed linear power spectrum which we need to calculate the
        # BAO damping and SPT integrals used in the Noda2017 model

        pk_smooth_lin = smooth_func(ks, pk_lin, method=self.smooth_type, om=om, h0=h0)
        pk_smooth_nonlin_0 = smooth_func(ks, pk_nonlin_0, method=self.smooth_type, om=om, h0=h0)
        pk_smooth_nonlin_z = smooth_func(ks, pk_nonlin_z, method=self.smooth_type, om=om, h0=h0)
        pk_smooth_spline = splrep(ks, pk_smooth_lin)

        # Sigma^2_dd,rs, Sigma^2_ss,rs (Noda2019 model)
        sigma_dd_rs = integrate.simps(pk_smooth_lin * (1.0 - j0 + 2.0 * j2), ks) / (6.0 * np.pi**2)
        sigma_ss_rs = integrate.simps(pk_smooth_lin * j2, ks) / (2.0 * np.pi**2)

        # I_00/P_sm,lin, I_01/P_sm,lin, I_02/P_sm,lin
        Pdd_spt = np.zeros(ks.shape)
        Pdt_spt = np.zeros(ks.shape)
        Ptt_spt = np.zeros(ks.shape)
        for k, kval in enumerate(ks):
            rvals = r[k, 0:]
            rx = np.outer(rvals, xs)
            y = kval * np.sqrt(-2.0 * rx.T + 1.0 + rvals**2)
            pk_smooth_interp = splev(y, pk_smooth_spline)
            index = np.where(np.logical_and(y < self.camb.k_min, y > self.camb.k_max))
            pk_smooth_interp[index] = 0.0
            IP0 = kval**2 * ((-10.0 * rx * xs + 7.0 * xs).T + 3.0 * rvals) / (y**2)
            IP1 = kval**2 * ((-6.0 * rx * xs + 7.0 * xs).T - rvals) / (y**2)
            Pdd_spt[k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP0 * IP0, xs, axis=0), rvals)
            Pdt_spt[k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP0 * IP1, xs, axis=0), rvals)
            Ptt_spt[k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP1 * IP1, xs, axis=0), rvals)
        Pdd_spt *= ks**3 / (392.0 * np.pi**2) / pk_smooth_lin
        Pdt_spt *= ks**3 / (392.0 * np.pi**2) / pk_smooth_lin
        Ptt_spt *= ks**3 / (392.0 * np.pi**2) / pk_smooth_lin

        # Add on k^2[J_00, J_01, J_11] to obtain P_sm,spt/P_sm,L - 1
        Pdd_spt += ks**2 * integrate.simps(pk_smooth_lin * J00, ks, axis=1) / (1008.0 * np.pi**2)
        Pdt_spt += ks**2 * integrate.simps(pk_smooth_lin * J01, ks, axis=1) / (1008.0 * np.pi**2)
        Ptt_spt += ks**2 * integrate.simps(pk_smooth_lin * J11, ks, axis=1) / (336.0 * np.pi**2)

        # Compute the non linear correction to the power spectra using the fitting formulae from Jennings2012
        growth_0, growth_z = self.get_growth_factor_Linder(om, 1.0e-4), self.get_growth_factor_Linder(om, self.camb.redshift)
        cfactor = (growth_z + growth_z**2 + growth_z**3) / (growth_0 + growth_0**2 + growth_0**3)
        Pdt_0 = (-12483.8 * np.sqrt(pk_smooth_nonlin_0) + 2.554 * pk_smooth_nonlin_0**2) / (1381.29 + 2.540 * pk_smooth_nonlin_0)
        Ptt_0 = (-12480.5 * np.sqrt(pk_smooth_nonlin_0) + 1.824 * pk_smooth_nonlin_0**2) / (2165.87 + 1.796 * pk_smooth_nonlin_0)
        Pdt_z = cfactor**2 * (Pdt_0 - pk_smooth_nonlin_0) + pk_smooth_nonlin_z
        Ptt_z = cfactor**2 * (Ptt_0 - pk_smooth_nonlin_0) + pk_smooth_nonlin_z

        Pdd_halofit = pk_smooth_nonlin_z / pk_smooth_lin - 1.0
        Pdt_halofit = Pdt_z / pk_smooth_lin - 1.0
        Ptt_halofit = Ptt_z / pk_smooth_lin - 1.0

        return {
            "sigma_dd_rs": sigma_dd_rs,
            "sigma_ss_rs": sigma_ss_rs,
            "Pdd_spt": Pdd_spt,
            "Pdt_spt": Pdt_spt,
            "Ptt_spt": Ptt_spt,
            "Pdd_halofit": Pdd_halofit,
            "Pdt_halofit": Pdt_halofit,
            "Ptt_halofit": Ptt_halofit,
        }

    @lru_cache(maxsize=1024)
    def get_growth_factor_Linder(self, omega_m, z, gamma=0.55):
        """
        Computes the unnormalised growth factor at redshift z given the present day value of omega_m. Uses the approximation
        from Linder2005 with fixed gamma

        :param omega_m: the matter density at the present day
        :param z: the redshift we want the matter density at
        :param gamma: the growth index. Default of 0.55 corresponding to LCDM.
        :return: the unnormalised growth factor at redshift z.
        """
        avals = np.logspace(-4.0, np.log10(1.0 / (1.0 + z)), 10000)
        f = Omega_m_z(omega_m, 1.0 / avals - 1.0) ** gamma
        integ = integrate.simps((f - 1.0) / avals, avals, axis=0)
        return np.exp(integ) / (1.0 + z)

    @lru_cache(maxsize=2)
    def get_extra(self):
        # Generate a grid of values for R1, R2, Imn and Jmn
        ks = self.camb.ks
        nx = 200
        xs = np.linspace(-0.999, 0.999, nx)
        r = np.outer(1.0 / ks, ks)

        J00 = (
            12.0 / r**2
            - 158.0
            + 100.0 * r**2
            - 42.0 * r**4
            + 3.0 * (r**2 - 1.0) ** 3 * (2.0 + 7.0 * r**2) / r**3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))
        )
        J01 = (
            24.0 / r**2
            - 202.0
            + 56.0 * r**2
            - 30.0 * r**4
            + 3.0 * (r**2 - 1.0) ** 3 * (4.0 + 5.0 * r**2) / r**3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))
        )
        J11 = (
            12.0 / r**2
            - 82.0
            + 4.0 * r**2
            - 6.0 * r**4
            + 3.0 * (r**2 - 1.0) ** 3 * (2.0 + r**2) / r**3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))
        )

        # We get NaNs in R1, R2 etc., when r = 1.0 (diagonals). We manually set these to the correct values.
        # We also get numerical issues for large/small r, so we set these manually to asymptotic limits
        J00[np.diag_indices(len(ks))] = -88.0
        J01[np.diag_indices(len(ks))] = -152.0
        J11[np.diag_indices(len(ks))] = -72.0
        index = np.where(r < 1.0e-3)
        J00[index] = -168.0
        J01[index] = -168.0
        J11[index] = -56.0
        index = np.where(r > 1.0e2)
        J00[index] = -97.6
        J01[index] = -200.0
        J11[index] = -100.8
        return r, xs, J00, J01, J11

    def validate_nonlinear_method(self):
        types = ["spt", "halofit"]
        if self.nonlinear_type in types:
            return True
        else:
            logging.getLogger("barry").error(f"Smoothing method is {self.nonlinear_type} and not in list {types}")
            return False

    @lru_cache(maxsize=32)
    def get_damping(self, growth, om, gamma):
        return np.exp(
            -np.outer(
                (1.0 + (2.0 + growth) * growth * self.mu**2) * self.get_pregen("sigma_dd_rs", om)
                + (growth * self.mu**2 * (self.mu**2 - 1.0)) * self.get_pregen("sigma_ss_rs", om),
                self.camb.ks**2,
            )
            / gamma
        )

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_par(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer((1.0 + (2.0 + growth) * growth) * ks**2, self.mu**2) * self.get_pregen("sigma_dd_rs", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_dd_perp(self, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(-np.outer(ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_dd_rs", om))

    @lru_cache(maxsize=4)
    def get_damping_aniso_ss_perp(self, growth, om, data_name=None):
        if data_name is None:
            ks = self.camb.ks
        else:
            ks = self.data_dict[data_name]["ks_input"]
        return np.exp(np.outer(growth * ks**2, 1.0 - self.mu**2) * self.get_pregen("sigma_ss_rs", om))

    @lru_cache(maxsize=32)
    def get_nonlinear(self, growth, om):
        return (
            self.get_pregen("Pdd_" + self.nonlinear_type, om),
            np.outer(2.0 * growth * self.mu**2, self.get_pregen("Pdt_" + self.nonlinear_type, om)),
            np.outer((growth * self.mu**2) ** 2, self.get_pregen("Ptt_" + self.nonlinear_type, om)),
        )

    @lru_cache(maxsize=32)
    def get_nonlinear_aniso(self, growth, om):
        return (
            splrep(self.camb.ks, self.get_pregen("Pdd_" + self.nonlinear_type, om)),
            splrep(self.camb.ks, 2.0 * growth * self.get_pregen("Pdt_" + self.nonlinear_type, om)),
            splrep(self.camb.ks, growth**2 * self.get_pregen("Ptt_" + self.nonlinear_type, om)),
        )

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
        self.add_param("gamma", r"$\gamma_{rec}$", 1.0, 8.0, 1.0)  # Describes the sharpening of the BAO post-reconstruction
        self.add_param("A", r"$A$", -10, 30.0, 10)  # Fingers-of-god damping

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None):
        """Computes the power spectrum model using the model from Noda et. al., 2019

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

        if self.isotropic:

            pk = [np.zeros(len(k))]

            kprime = k / p["alpha"]

            fog = np.exp(-p["A"] * ks**2)
            pk_smooth = p["b{0}"] ** 2 * pk_smooth_lin * fog

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            # Lets round some things for the sake of numerical speed (to hit the cache more often
            om = np.round(p["om"], decimals=5)
            growth = np.round(p["beta"] * p["b{0}"], decimals=5)
            gamma = np.round(p["gamma"], decimals=5)

            if self.recon:
                kaiser_prefac = 1.0 + np.outer(p["beta"] * self.mu**2, 1.0 - self.camb.smoothing_kernel)
            else:
                kaiser_prefac = 1.0 + np.tile(p["beta"] * self.mu**2, (len(ks), 1)).T

            # Compute the non-linear correction to the smooth power spectrum
            p_dd, p_dt, p_tt = self.get_nonlinear(growth, om)
            pk_nonlinear = p_dd + p_dt / p["b{0}"] + p_tt / p["b{0}"] ** 2

            # Integrate over mu
            if smooth:
                pk1d = integrate.simps(pk_smooth * (kaiser_prefac**2 + pk_nonlinear), self.mu, axis=0)
            else:
                # Compute the BAO damping/propagator
                propagator = self.get_damping(growth, om, gamma)
                pk1d = integrate.simps(pk_smooth * ((1.0 + pk_ratio * propagator) * kaiser_prefac**2 + pk_nonlinear), self.mu, axis=0)

            poly = np.zeros((1, len(k)))
            pk[0] = splev(kprime, splrep(ks, pk1d))

        else:
            epsilon = np.round(p["epsilon"], decimals=5)
            kprime = np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.get_muprime(epsilon)
            fog = np.exp(-p["A"] * muprime**2 * kprime**2)
            pk_smooth = p["b{0}"] ** 2 * splev(kprime, splrep(ks, pk_smooth_lin)) * fog

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            # Lets round some things for the sake of numerical speed
            om = np.round(p["om"], decimals=5)
            growth = np.round(p["beta"] * p["b{0}"], decimals=5)
            gamma = np.round(p["gamma"], decimals=5)

            if self.recon:
                sprime = splev(kprime, splrep(ks, self.camb.smoothing_kernel))
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - sprime)
            else:
                kaiser_prefac = 1.0 + p["beta"] * muprime**2

            # Compute the non-linear correction to the smooth power spectrum
            p_dd_spline, p_dt_spline, p_tt_spline = self.get_nonlinear_aniso(growth, om)
            pk_nonlinear = (
                splev(kprime, p_dd_spline)
                + muprime**2 * splev(kprime, p_dt_spline) / p["b{0}"]
                + muprime**4 * splev(kprime, p_tt_spline) / p["b{0}"] ** 2
            )

            if smooth:
                pk2d = pk_smooth * (kaiser_prefac**2 + pk_nonlinear)
            else:
                # Compute the BAO damping
                power_par = 1.0 / (p["alpha"] ** 2 * (1.0 + epsilon) ** 4 * gamma)
                power_perp = (1.0 + epsilon) ** 2 / (p["alpha"] ** 2 * gamma)
                propagator = (
                    self.get_damping_aniso_dd_par(growth, om, data_name=data_name) ** power_par
                    * self.get_damping_aniso_dd_perp(om, data_name=data_name) ** power_perp
                    * self.get_damping_aniso_ss_perp(growth, om, data_name=data_name) ** (power_perp * muprime**2)
                )
                pk2d = pk_smooth * ((1.0 + splev(kprime, splrep(ks, pk_ratio)) * propagator) * kaiser_prefac**2 + pk_nonlinear)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]
            poly = np.zeros((1, 5, len(k)))

        return kprime, pk, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12
    from barry.postprocessing import BAOExtractor
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    """
    postprocess = BAOExtractor(147.6)

    # TODO: Make sanity check make sense for BAOEXtracted measurements. Plot is not really useful.
    """

    print("Checking isotropic mock mean")
    dataset = PowerSpectrum_SDSS_DR12(isotropic=True, recon="iso")
    model = PowerNoda2019(recon=dataset.recon, marg=None, isotropic=dataset.isotropic, correction=Correction.HARTLAP)
    model.sanity_check(dataset)

    print("Checking anisotropic mock mean")
    dataset = PowerSpectrum_SDSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2, 4])
    model = PowerNoda2019(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg=None,
        correction=Correction.HARTLAP,
    )
    model.sanity_check(dataset)
