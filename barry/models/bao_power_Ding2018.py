import logging
from functools import lru_cache

import numpy as np
from scipy.interpolate import splev, splrep
from scipy import integrate
from barry.models.bao_power import PowerSpectrumFit


class PowerDing2018(PowerSpectrumFit):
    """ Model from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(self, fix_params=["om", "f"], smooth_type="hinton2017", recon=False, name="Pk Ding 2018", postprocess=None, smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(fix_params=fix_params, smooth_type=smooth_type, name=name, postprocess=postprocess, smooth=smooth, correction=correction)

        self.fit_omega_m = fix_params is None or "om" not in fix_params
        self.fit_growth = fix_params is None or "f" not in fix_params
        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.smoothing_kernel = None

    @lru_cache(maxsize=32)
    def get_pt_data(self, om):
        return self.PT.get_data(om=om)

    @lru_cache(maxsize=8192)
    def get_damping_dd(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pt_data(om)["sigma_dd_nl"])

    @lru_cache(maxsize=8192)
    def get_damping_sd(self, growth, om):
        return np.exp(-np.outer(1.0 + growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pt_data(om)["sigma_dd_nl"])

    @lru_cache(maxsize=32)
    def get_damping_ss(self, om):
        return np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.get_pt_data(om)["sigma_ss_nl"])

    @lru_cache(maxsize=8192)
    def get_damping(self, growth, om):
        return np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, self.camb.ks ** 2) * self.get_pt_data(om)["sigma_nl"])

    def set_data(self, data):
        super().set_data(data)
        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        if self.recon:
            self.smoothing_kernel = np.exp(-self.camb.ks ** 2 * self.recon_smoothing_scale ** 2 / 2.0)

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0, 5.0)  # Non-linear galaxy bias
        self.add_param("a1", r"$a_1$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -50000.0, 50000.0, 0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -1000.0, 1000.0, 0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -10.0, 10.0, 0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p, smooth=False):
        """ Computes the power spectrum model at k/alpha using the Ding et. al., 2018 EFT0 model
        
        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute
        p : dict
            dictionary of parameter names to their values
            
        Returns
        -------
        array
            pk_final - The power spectrum at the dilated k-values
        
        """
        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])

        # Compute the growth rate depending on what we have left as free parameters
        if self.fit_growth:
            growth = p["f"]
        else:
            growth = p["om"]**0.55

        # Lets round some things for the sake of numerical speed
        om = np.round(p["om"], decimals=5)
        growth = np.round(growth, decimals=5)

        # Compute the BAO damping
        if self.recon:
            damping_dd = self.get_damping_dd(growth, om)
            damping_sd = self.get_damping_sd(growth, om)
            damping_ss = self.get_damping_ss(om)
        else:
            damping = self.get_damping(growth, om)

        # Compute the propagator
        if self.recon:
            smooth_prefac = np.tile(self.smoothing_kernel/p["b"], (self.nmu, 1))
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (self.nmu, 1))
            kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth/p["b"]*self.mu**2, 1.0-self.smoothing_kernel) + bdelta_prefac
            propagator = (1**2 - (bdelta_prefac/kaiser_prefac)**2)*damping_dd + 2.0*smooth_prefac*damping_sd/kaiser_prefac + smooth_prefac**2*damping_ss/kaiser_prefac**2
        else:
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (self.nmu, 1))
            kaiser_prefac = 1.0 + np.tile(growth/p["b"]*self.mu**2, (len(ks), 1)).T + bdelta_prefac
            propagator = (1**2 - (bdelta_prefac/kaiser_prefac)**2)*damping

        # Compute the smooth model
        fog = 1.0/(1.0 + np.outer(self.mu**2, ks**2*p["sigma_s"]**2/2.0))**2
        pk_smooth = kaiser_prefac**2 *p["b"]**2*pk_smooth_lin*fog

        # Polynomial shape
        if self.recon:
            shape = p["a1"] * ks**2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

        # Integrate over mu
        if smooth:
            pk1d = integrate.simps((pk_smooth + shape)*(1.0 + 0.0 * pk_ratio*propagator), self.mu, axis=0)
        else:
            pk1d = integrate.simps((pk_smooth + shape)*(1.0 + pk_ratio*propagator), self.mu, axis=0)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d))

        return pk_final


if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    model_pre = PowerDing2018(recon=False)
    model_post = PowerDing2018(recon=True)

    from barry.datasets.mock_power import PowerSpectrum_SDSS_DR12_Z061_NGC
    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC()
    data = dataset.get_data()
    model_pre.set_data(data)
    model_post.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_s": 10.0, "b": 1.6, "b_delta": 1, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    import timeit
    n = 200

    def test():
        model_post.get_likelihood(p, data[0])

    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data[0]["ks"]
        pk = data[0]["pk"]
        pk2 = model_pre.get_model(p, data[0])
        model_pre.smooth_type = "eh1998"
        pk3 = model_pre.get_model(p, data[0])
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data[0]["cov"])), fmt="o", c='k', label="Data")
        plt.plot(ks, pk2, '.', c='r', label="hinton2017")
        plt.plot(ks, pk3, '+', c='b', label="eh1998")
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        model_pre.smooth_type = "hinton2017"
        pk_smooth_lin, _ = model_pre.compute_basic_power_spectrum(p["om"])
        pk_smooth_interp = splev(data[0]["ks_input"], splrep(model_pre.camb.ks, pk_smooth_lin))
        pk_smooth_lin_windowed, mask = model_pre.adjust_model_window_effects(pk_smooth_interp, data[0])
        pk2 = model_pre.get_model(p, data[0])
        pk3 = model_post.get_model(p, data[0])
        import matplotlib.pyplot as plt
        plt.plot(ks, pk2/pk_smooth_lin_windowed[mask], '.', c='r', label="pre-recon")
        plt.plot(ks, pk3/pk_smooth_lin_windowed[mask], '+', c='b', label="post-recon")
        plt.xlabel("k")
        plt.ylabel(r"$P(k)/P_{sm}(k)$")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.4, 3.0)
        plt.legend()
        plt.show()