import logging
import numpy as np
from scipy.interpolate import splev, splrep
from barry.models.bao_power import PowerSpectrumFit


class PowerBeutler2017(PowerSpectrumFit):
    """ P(k) model inspired from Beutler 2017.

    See https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3409B for details.

    """

    def __init__(self, name="Pk Beutler 2017", fix_params=("om"), smooth_type="hinton2017", recon=False, postprocess=None, smooth=False, correction=None):
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction)

        self.recon = recon

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.01, 20.0, 10.0)  # BAO damping
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 20.0, 10.0)  # Fingers-of-god damping
        self.add_param("a1", r"$a_1$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -1000.0, 1000.0, 0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -10.0, 10.0, 0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p, smooth=False):
        """ Computes the power spectrum for the Beutler et. al., 2017 model at k/alpha
        
        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute
        p : dict
            dictionary of parameter names to their values
        smooth : bool, optional
            Whether to return a smooth model or not. Defaults to False
            
        Returns
        -------
        array
            pk_final - The power spectrum at the dilated k-values
        
        """

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # Compute the propagator
        C = np.exp(-0.5 * ks ** 2 * p["sigma_nl"] ** 2)

        # Compute the smooth model
        fog = 1.0 / (1.0 + ks ** 2 * p["sigma_s"] ** 2 / 2.0) ** 2
        pk_smooth = p["b"] ** 2 * pk_smooth_lin * fog

        # Polynomial shape
        if self.recon:
            shape = p["a1"] * ks ** 2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

        if smooth:
            pk_final = splev(k / p["alpha"], splrep(ks, pk_smooth + shape))
        else:
            pk_final = splev(k / p["alpha"], splrep(ks, (pk_smooth + shape) * (1.0 + pk_ratio * C)))

        return pk_final


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    recon = True
    model1 = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}")
    model_smooth = PowerBeutler2017(recon=recon, name=f"Beutler2017, recon={recon}", smooth=True)

    from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC

    dataset1 = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=recon)
    data1 = dataset1.get_data()
    model1.set_data(data1)

    # First comparison - the actual recon data
    if False:
        p, minv = model1.optimize()
        model_smooth.set_data(data1)
        p2, minv2 = model_smooth.optimize()
        print(p)
        print(minv)
        model1.plot(p, smooth_params=p2)

    if True:
        import timeit

        n = 200
        p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 5, "sigma_s": 10.0, "b": 1.6, "b_delta": 1, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

        def test():
            model1.get_likelihood(p, data1[0])

        print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if False:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model.get_model(data, p)
        model.smooth_type = "eh1998"
        pk3 = model.get_model(data, p)
        import matplotlib.pyplot as plt

        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c="k", label="Data")
        plt.plot(ks, pk2, ".", c="r", label="hinton2017")
        plt.plot(ks, pk3, "+", c="b", label="eh1998")
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

        model.smooth_type = "hinton2017"
        pk_smooth_lin, _ = model.compute_basic_power_spectrum(p["om"])
        pk_smooth_interp = splev(data["ks_input"], splrep(model.camb.ks, pk_smooth_lin))
        pk_smooth_lin_windowed, mask = model.adjust_model_window_effects(pk_smooth_interp)
        pk2 = model.get_model(data, p)
        import matplotlib.pyplot as plt

        plt.plot(ks, pk2 / pk_smooth_lin_windowed[mask], ".", c="r", label="pre-recon")
        plt.xlabel("k")
        plt.ylabel(r"$P(k)/P_{sm}(k)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.4, 3.0)
        plt.legend()
        plt.show()
