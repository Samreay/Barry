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
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        dilate_smooth=True,
        n_poly=5,
        n_data=1,
        data_share_bias=False,
        data_share_poly=False,
    ):

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
                for ip in range(self.n_poly):
                    self.add_param(f"a{{{pole}}}_{{{ip+1}}}_{{{i+1}}}", f"$a_{{{pole},{ip+1},{i+1}}}$", -20000.0, 20000.0, 0)

    def compute_power_spectrum(self, k, p, smooth=False, for_corr=False, data_name=None, nopoly=False):
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
            if "b{0}" not in p:
                p = self.deal_with_ndata(p, 0)

        if self.isotropic:
            pk = [np.zeros(len(k))]
            kprime = k if for_corr else k / p["alpha"]
            if self.dilate_smooth:
                pk_smooth = splev(kprime, splrep(ks, pk_smooth_lin)) / (1.0 + kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
            else:
                pk_smooth = splev(k, splrep(ks, pk_smooth_lin)) / (1.0 + k**2 * p["sigma_s"] ** 2 / 2.0) ** 2
            if not for_corr:
                pk_smooth *= p["b{0}"]

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            if smooth:
                propagator = np.ones(len(kprime))
            else:
                # Compute the propagator
                C = np.exp(-0.5 * kprime**2 * p["sigma_nl"] ** 2)
                propagator = 1.0 + splev(kprime, splrep(ks, pk_ratio)) * C
            prefac = np.ones(len(kprime)) if smooth else propagator

            if for_corr or nopoly:
                poly = None
                pk[0] = pk_smooth * propagator
            else:
                shape, poly = self.add_poly(k, k, p, prefac, pk_smooth)
                if not self.marg:
                    pk[0] = (pk_smooth + shape) * propagator

        else:

            epsilon = 0 if for_corr else p["epsilon"]
            kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
            muprime = self.mu if for_corr else self.get_muprime(epsilon)
            if self.dilate_smooth:
                fog = 1.0 / (1.0 + muprime**2 * kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
                reconfac = splev(kprime, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
                pk_smooth = kaiser_prefac**2 * splev(kprime, splrep(ks, pk_smooth_lin))
            else:
                ktile = np.tile(k, (self.nmu, 1)).T
                fog = 1.0 / (1.0 + muprime**2 * ktile**2 * p["sigma_s"] ** 2 / 2.0) ** 2
                reconfac = splev(ktile, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
                kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
                pk_smooth = kaiser_prefac**2 * splev(ktile, splrep(ks, pk_smooth_lin))

            if not for_corr:
                pk_smooth *= p["b{0}"]

            # Volume factor
            pk_smooth /= p["alpha"] ** 3

            # Compute the propagator
            if smooth:
                pk2d = pk_smooth * fog
            else:
                C = np.exp(-0.5 * kprime**2 * (muprime**2 * p["sigma_nl_par"] ** 2 + (1.0 - muprime**2) * p["sigma_nl_perp"] ** 2))
                pk2d = pk_smooth * (fog + splev(kprime, splrep(ks, pk_ratio)) * C)

            pk0, pk2, pk4 = self.integrate_mu(pk2d)

            # Polynomial shape
            pk = [pk0, np.zeros(len(k)), pk2, np.zeros(len(k)), pk4, np.zeros(len(k))]

            if for_corr or nopoly:
                poly = None
                kprime = k
            else:
                shape, poly = self.add_poly(k, k, p, np.ones(len(k)), pk)
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
        PowerSpectrum_DESI_KP4,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    dataset = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.00,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=1,
    )
    data = dataset.get_data()

    model = PowerBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=5,
    )
    model.set_default("sigma_nl_par", 5.4, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_nl_perp", 1.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_s", 0.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.kvals, model.pksmooth, model.pkratio = pktemplate.T

    model.sanity_check(dataset)
