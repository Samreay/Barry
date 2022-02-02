import numpy as np
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep


class PowerBeutler2017_3poly(PowerSpectrumFit):
    """P(k) model inspired from Beutler 2017, but with only 3 polynomial terms.

    See https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3409B for details.

    """

    def __init__(
        self,
        name="Pk Beutler 2017 3poly",
        fix_params=("om",),
        smooth_type="hinton2017",
        recon=None,
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
    ):

        if isotropic:
            poly_poles = [0]
        if marg is not None:
            fix_params = list(fix_params)
            fix_params.extend(["b"])
            for pole in poly_poles:
                fix_params.extend([f"a{{{pole}}}_1", f"a{{{pole}}}_2", f"a{{{pole}}}_3"])

        self.poly_poles = poly_poles

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
        )
        if self.marg:
            self.set_default("b", 1.0)
            for pole in self.poly_poles:
                self.set_default(f"a{{{pole}}}_1", 0.0)
                self.set_default(f"a{{{pole}}}_2", 0.0)
                self.set_default(f"a{{{pole}}}_3", 0.0)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.01, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, 0.5)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for pole in self.poly_poles:
            self.add_param(f"a{{{pole}}}_1", f"$a_{{{pole},1}}$", -20000.0, 20000.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param(f"a{{{pole}}}_2", f"$a_{{{pole},2}}$", -20000.0, 20000.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param(f"a{{{pole}}}_3", f"$a_{{{pole},3}}$", -5000.0, 5000.0, 0)  # Monopole Polynomial marginalisation 3

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None):
        """Computes the power spectrum model using the Beutler et. al., 2017 method

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

        # We split for isotropic and anisotropic here for consistency with our previous isotropic convention, which
        # differs from our implementation of the Beutler2017 isotropic model quite a bit. This results in some duplication
        # of code and a few nested if statements, but it's perhaps more readable and a little faster (because we only
        # need one interpolation for the whole isotropic monopole, rather than separately for the smooth and wiggle components)

        if self.isotropic:
            pk = [np.zeros(len(k))]
            kprime = k if for_corr else k / p["alpha"]
            fog = 1.0 / (1.0 + kprime ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            pk_smooth = splev(kprime, splrep(ks, pk_smooth_lin)) * fog
            if not for_corr:
                pk_smooth *= p["b"]
                if self.recon:
                    shape = p["a{0}_1"] * kprime ** 2 + p["a{0}_2"] + p["a{0}_3"] / kprime
                else:
                    shape = p["a{0}_1"] * kprime + p["a{0}_2"] + p["a{0}_3"] / kprime

            if smooth:
                pk[0] = pk_smooth if for_corr else pk_smooth + shape
            else:
                # Compute the propagator
                C = np.exp(-0.5 * ks ** 2 * p["sigma_nl"] ** 2)
                propagator = splev(kprime, splrep(ks, (1.0 + pk_ratio * C)))
                pk[0] = pk_smooth * propagator if for_corr else (pk_smooth + shape) * propagator

            poly = np.zeros((1, len(k)))
            if self.marg:
                prefac = np.ones(len(kprime)) if smooth else propagator
                poly = prefac * [pk_smooth, kprime, np.ones(len(kprime)), 1.0 / kprime]
                if self.recon:
                    poly[1] *= kprime

        else:

            epsilon = 0 if for_corr else p["epsilon"]
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.mu if for_corr else self.get_muprime(epsilon)
            fog = 1.0 / (1.0 + muprime ** 2 * kprime ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            if self.recon_type.lower() == "iso":
                kaiser_prefac = 1.0 + p["beta"] * muprime ** 2 * (1.0 - splev(kprime, splrep(self.camb.ks, self.camb.smoothing_kernel)))
            else:
                kaiser_prefac = 1.0 + p["beta"] * muprime ** 2
            pk_smooth = kaiser_prefac ** 2 * splev(kprime, splrep(ks, pk_smooth_lin)) * fog
            if not for_corr:
                pk_smooth *= p["b"]

            # Compute the propagator
            if smooth:
                pk2d = pk_smooth
            else:
                C = np.exp(-0.5 * kprime ** 2 * (muprime ** 2 * p["sigma_nl_par"] ** 2 + (1.0 - muprime ** 2) * p["sigma_nl_perp"] ** 2))
                pk2d = pk_smooth * (1.0 + splev(kprime, splrep(ks, pk_ratio)) * C)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

            # Polynomial shape
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4]

            if for_corr:
                poly = None
                kprime = k
            else:
                npoly = 3
                if self.marg:
                    poly = np.zeros((npoly * len(self.poly_poles) + 1, 5, len(k)))
                    poly[0, :, :] = pk
                    for i, pole in enumerate(self.poly_poles):
                        if self.recon:
                            poly[npoly * i + 1 : npoly * (i + 1) + 1, pole] = [k, np.ones(len(k)), 1.0 / k]
                        else:
                            poly[npoly * i + 1 : npoly * (i + 1) + 1, pole] = [k, np.ones(len(k)), 1.0 / k]

                    pk = [np.zeros(len(k))] * 5

                else:
                    poly = np.zeros((1, 5, len(k)))
                    for pole in self.poly_poles:
                        if self.recon:
                            pk[pole] += p[f"a{{{pole}}}_1"] * k + p[f"a{{{pole}}}_2"] + p[f"a{{{pole}}}_3"] / k
                        else:
                            pk[pole] += p[f"a{{{pole}}}_1"] * k + p[f"a{{{pole}}}_2"] + p[f"a{{{pole}}}_3"] / k

        return kprime, pk, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    print("Checking isotropic mock mean")
    dataset = PowerSpectrum_SDSS_DR12(isotropic=True, recon="iso")
    model = PowerBeutler2017_3poly(recon=dataset.recon, marg="full", isotropic=dataset.isotropic, correction=Correction.HARTLAP)
    model.sanity_check(dataset)

    print("Checking anisotropic mock mean")
    dataset = PowerSpectrum_SDSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2, 4])
    model = PowerBeutler2017_3poly(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2, 4],
        correction=Correction.HARTLAP,
    )
    model.sanity_check(dataset)
