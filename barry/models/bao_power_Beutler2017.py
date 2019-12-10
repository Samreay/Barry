import numpy as np

from barry.models.bao_power import PowerSpectrumFit
from scipy.interpolate import splev, splrep
from scipy import integrate


class PowerBeutler2017(PowerSpectrumFit):
    """ P(k) model inspired from Beutler 2017.

    See https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3409B for details.

    """

    def __init__(
        self, name="Pk Beutler 2017", fix_params=("om"), smooth_type="hinton2017", recon=False, postprocess=None, smooth=False, correction=None, isotropic=True
    ):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(
            name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction, isotropic=isotropic
        )

    def declare_parameters(self):
        super().declare_parameters()
        # print(self.isotropic)
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.01, 20.0, 10.0)  # BAO damping
            self.add_param("a1", r"$a_1$", -10000.0, 30000.0, 0)  # Polynomial marginalisation 1
            self.add_param("a2", r"$a_2$", -20000.0, 10000.0, 0)  # Polynomial marginalisation 2
            self.add_param("a3", r"$a_3$", -1000.0, 5000.0, 0)  # Polynomial marginalisation 3
            self.add_param("a4", r"$a_4$", -200.0, 200.0, 0)  # Polynomial marginalisation 4
            self.add_param("a5", r"$a_5$", -3.0, 3.0, 0)  # Polynomial marginalisation 5
        else:
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.01, 20.0, 10.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.01, 20.0, 10.0)  # BAO damping perpendicular to LOS
            self.add_param("a0_1", r"$a0_1$", -10000.0, 30000.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param("a0_2", r"$a0_2$", -20000.0, 10000.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param("a0_3", r"$a0_3$", -1000.0, 5000.0, 0)  # Monopole Polynomial marginalisation 3
            self.add_param("a0_4", r"$a0_4$", -200.0, 200.0, 0)  # Monopole Polynomial marginalisation 4
            self.add_param("a0_5", r"$a0_5$", -3.0, 3.0, 0)  # Monopole Polynomial marginalisation 5
            self.add_param("a2_1", r"$a2_1$", -10000.0, 30000.0, 0)  # Monopole Polynomial marginalisation 1
            self.add_param("a2_2", r"$a2_2$", -20000.0, 10000.0, 0)  # Monopole Polynomial marginalisation 2
            self.add_param("a2_3", r"$a2_3$", -1000.0, 5000.0, 0)  # Monopole Polynomial marginalisation 3
            self.add_param("a2_4", r"$a2_4$", -200.0, 200.0, 0)  # Monopole Polynomial marginalisation 4
            self.add_param("a2_5", r"$a2_5$", -3.0, 3.0, 0)  # Monopole Polynomial marginalisation 5

    def compute_power_spectrum(self, p, smooth=False, shape=True):
        """ Computes the power spectrum multipoles for the Beutler et. al., 2017 model at dilated k values

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
        pk0 : np.ndarray
            the model monopole interpolated using the dilation scales.
        pk2 : np.ndarray
            the model quadrupole interpolated using the dilation scales. Will be 'None' if the model is isotropic

        """

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # We split for isotropic and anisotropic here for consistency with our previous isotropic convention, which
        # differs from our implementation of the Beutler2017 isotropic model quite a bit. This results in some duplication
        # of code and a few nested if statements, but it's perhaps more readable and a little faster (because we only
        # need one interpolation for the whole isotropic monopole, rather than separately for the smooth and wiggle components)

        if self.isotropic:
            fog = 1.0 / (1.0 + ks ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            pk_smooth = p["b"] * pk_smooth_lin * fog

            # Polynomial shape
            if shape:
                if self.recon:
                    shape = p["a1"] * ks ** 2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
                else:
                    shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
            else:
                shape = 0

            kprime = ks / p["alpha"]

            if smooth:
                pk0 = splev(kprime, splrep(ks, pk_smooth + shape))
            else:
                # Compute the propagator
                C = np.exp(-0.5 * kprime ** 2 * p["sigma_nl"] ** 2)
                pk0 = splev(kprime, splrep(ks, (pk_smooth + shape) * (1.0 + pk_ratio * C)))

            pk2 = None

        else:
            kprime = self.get_kprime(p["alpha"], p["epsilon"])
            muprime = self.get_muprime(p["alpha"], p["epsilon"])
            sprime = splev(kprime, splrep(ks, self.camb.smoothing_kernel))

            fog = 1.0 / (1.0 + muprime ** 2 * kprime ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
            kaiser_prefac = (p["b"] ** 2 + p["f"] * muprime ** 2 * (1.0 - sprime)) ** 2
            pk_smooth = kaiser_prefac * pk_smooth_lin * fog

            # Compute the propagator
            C = np.exp(-0.5 * kprime ** 2 * (muprime ** 2 * p["sigma_nl_par"] ** 2 + (1.0 - muprime ** 2) * p["sigma_nl_perp"] ** 2))
            pk2d = pk_smooth * (1.0 + pk_ratio * C)

            pk0 = integrate.simps(pk2d, self.mu, axis=1)
            pk2 = 2.5 * (3.0 * integrate.simps(pk2d * self.mu ** 2, self.mu, axis=1) - pk0)

            # Polynomial shape
            if shape:
                if self.recon:
                    shape0 = p["a0_1"] * ks ** 2 + p["a0_2"] + p["a0_3"] / ks + p["a0_4"] / (ks * ks) + p["a0_5"] / (ks ** 3)
                    shape2 = p["a2_1"] * ks ** 2 + p["a2_2"] + p["a2_3"] / ks + p["a2_4"] / (ks * ks) + p["a2_5"] / (ks ** 3)
                else:
                    shape0 = p["a0_1"] * ks + p["a0_2"] + p["a0_3"] / ks + p["a0_4"] / (ks * ks) + p["a0_5"] / (ks ** 3)
                    shape2 = p["a2_1"] * ks + p["a2_2"] + p["a2_3"] / ks + p["a2_4"] / (ks * ks) + p["a2_5"] / (ks ** 3)
            else:
                shape0 = 0
                shape2 = 0

            pk0 += shape0
            pk2 += shape2

        return ks, pk0, pk2


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12_Z061_NGC
    from barry.config import setup_logging

    setup_logging()

    print("Checking pre-recon")
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=False)
    model_pre = PowerBeutler2017(recon=False)
    model_pre.sanity_check(dataset)

    print("Checking post-recon")
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True)
    model_post = PowerBeutler2017(recon=True)
    model_post.sanity_check(dataset)
