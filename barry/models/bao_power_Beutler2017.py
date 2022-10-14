import numpy as np
from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep


class PowerBeutler2017(PowerSpectrumFit):
    """P(k) model inspired from Beutler 2017.

    See https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3409B for details.

    """

    def __init__(
        self,
        name="Pk Beutler 2017",
        fix_params=("om",),
        smooth_type=None,
        recon=None,
        postprocess=None,
        smooth=False,
        correction=None,
        isotropic=True,
        poly_poles=(0, 2),
        marg=None,
        dilate_smooth=True,
        n_poly=5,
        n_data=1,
        data_share_bias=False,
        data_share_poly=False,
    ):

        if n_poly not in [0, 3, 5]:
            raise NotImplementedError("Models require n_poly to be 0, 3 or 5 polynomial terms per multipole")

        self.dilate_smooth = dilate_smooth

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
            n_poly=n_poly,
            n_data=n_data,
            data_share_bias=data_share_bias,
            data_share_poly=data_share_poly,
        )

        self.set_marg(fix_params, poly_poles, n_poly, do_bias=True)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.0, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.0, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.0, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for i in range(self.n_data_poly):
            for pole in self.poly_poles:
                if self.n_poly >= 3:
                    self.add_param(f"a{{{pole}}}_1_{{{i+1}}}", f"$a_{{{pole},1,{i+1}}}$", -20000.0, 20000.0, 0)
                    self.add_param(f"a{{{pole}}}_2_{{{i+1}}}", f"$a_{{{pole},2,{i+1}}}$", -20000.0, 20000.0, 0)
                    self.add_param(f"a{{{pole}}}_3_{{{i+1}}}", f"$a_{{{pole},3,{i+1}}}$", -5000.0, 5000.0, 0)
                if self.n_poly == 5:
                    self.add_param(f"a{{{pole}}}_4_{{{i+1}}}", f"$a_{{{pole},4,{i+1}}}$", -200.0, 200.0, 0)
                    self.add_param(f"a{{{pole}}}_5_{{{i+1}}}", f"$a_{{{pole},5,{i+1}}}$", -3.0, 3.0, 0)

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

        if not for_corr:
            if "b" not in p:
                p = self.deal_with_ndata(p, 0)

        if self.isotropic:
            pk = [np.zeros(len(k))]
            kprime = k if for_corr else k / p["alpha"]
            if self.dilate_smooth:
                pk_smooth = splev(kprime, splrep(ks, pk_smooth_lin)) / (1.0 + kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
            else:
                pk_smooth = splev(k, splrep(ks, pk_smooth_lin)) / (1.0 + k**2 * p["sigma_s"] ** 2 / 2.0) ** 2
            if not for_corr:
                pk_smooth *= p["b"]

            if smooth:
                propagator = np.ones(len(kprime))
            else:
                # Compute the propagator
                C = np.exp(-0.5 * kprime**2 * p["sigma_nl"] ** 2)
                propagator = 1.0 + splev(kprime, splrep(ks, pk_ratio)) * C
            prefac = np.ones(len(kprime)) if smooth else propagator

            if for_corr:
                poly = None
                pk[0] = pk_smooth * propagator
            else:
                shape, poly = (
                    self.add_three_poly(k, k, p, prefac, pk_smooth) if self.n_poly == 3 else self.add_five_poly(k, k, p, prefac, pk_smooth)
                )
                pk[0] = (pk_smooth + shape) * propagator

        else:

            epsilon = 0 if for_corr else p["epsilon"]
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.mu if for_corr else self.get_muprime(epsilon)
            if self.dilate_smooth:
                fog = 1.0 / (1.0 + muprime**2 * kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
                reconfac = splev(kprime, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
                pk_smooth = kaiser_prefac**2 * splev(kprime, splrep(ks, pk_smooth_lin)) * fog
            else:
                ktile = np.tile(k, (self.nmu, 1)).T
                fog = 1.0 / (1.0 + muprime**2 * ktile**2 * p["sigma_s"] ** 2 / 2.0) ** 2
                reconfac = splev(ktile, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
                pk_smooth = kaiser_prefac**2 * splev(ktile, splrep(ks, pk_smooth_lin)) * fog

            if not for_corr:
                pk_smooth *= p["b"]

            # Compute the propagator
            if smooth:
                pk2d = pk_smooth
            else:
                C = np.exp(-0.5 * kprime**2 * (muprime**2 * p["sigma_nl_par"] ** 2 + (1.0 - muprime**2) * p["sigma_nl_perp"] ** 2))
                pk2d = pk_smooth * (1.0 + splev(kprime, splrep(ks, pk_ratio)) * C)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

            # Polynomial shape
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]

            if for_corr:
                poly = None
                kprime = k
            else:
                shape, poly = (
                    self.add_three_poly(k, k, p, np.ones(len(k)), pk)
                    if self.n_poly == 3
                    else self.add_five_poly(k, k, p, np.ones(len(k)), pk)
                )
                if self.marg:
                    pk = [np.zeros(len(k))] * 6
                else:
                    for pole in self.poly_poles:
                        pk[pole] += shape[pole]

        return kprime, pk, poly


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import (
        PowerSpectrum_SDSS_DR12,
        PowerSpectrum_eBOSS_LRGpCMASS,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    print("Checking isotropic mock mean")
    dataset = PowerSpectrum_eBOSS_LRGpCMASS(realisation=None, isotropic=True, recon=None, galactic_cap="both")
    model = PowerBeutler2017(
        recon=dataset.recon,
        marg="full",
        isotropic=dataset.isotropic,
        correction=Correction.HARTLAP,
        n_data=dataset.ndata,
        n_poly=3,
    )
    model.sanity_check(dataset)

    print("Checking anisotropic mock mean")
    dataset = PowerSpectrum_eBOSS_LRGpCMASS(realisation=None, isotropic=False, fit_poles=[0, 2, 4], recon="iso", galactic_cap="both")
    model = PowerBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2, 4],
        correction=Correction.HARTLAP,
        n_data=dataset.ndata,
        n_poly=3,
    )
    model.sanity_check(dataset)
