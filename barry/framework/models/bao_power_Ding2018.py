import logging
import numpy as np
from scipy.interpolate import splev, splrep
import sys
sys.path.append("../../..")
from barry.framework.models.bao_power import PowerSpectrumFit
from barry.framework.cosmology.PT_generator import PTGenerator

class PowerDing2018(PowerSpectrumFit):

    def __init__(self, fit_omega_m=False, fit_growth=False, smooth_type="hinton2017", recon=False, recon_smoothing_scale=10.0, name="BAO Power Spectrum Ding 2018 Fit"):
        self.recon = recon
        self.recon_smoothing_scale = recon_smoothing_scale
        self.fit_growth = fit_growth
        super().__init__(fit_omega_m=fit_omega_m, smooth_type=smooth_type, name=name)

        self.PT = PTGenerator(self.camb, smooth_type=self.smooth_type, recon_smoothing_scale=self.recon_smoothing_scale)
        if not self.fit_omega_m:
            _, _, _, self.sigma_nl, self.sigma_dd_nl, self.sigma_sd_nl, self.sigma_ss_nl, _, _, _, _ = self.PT.get_data(om=self.omega_m)

    def declare_parameters(self):
        super().declare_parameters()
        if self.fit_growth:
            self.add_param("f", r"$f$", 0.01, 1.0)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0)  # Non-linear galaxy bias
        self.add_param("a1", r"$a_1$", -50000.0, 50000.0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -50000.0, 50000.0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -50000.0, 50000.0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -1000.0, 1000.0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -10.0, 10.0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p):
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

        from scipy import integrate

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(ks, p)
        if self.fit_omega_m:
            _, _, _, sigma_nl, sigma_dd_nl, sigma_sd_nl, sigma_ss_nl, _, _, _, _ = self.PT.get_data(om=p["om"])
        else:
            sigma_nl, sigma_dd_nl, sigma_sd_nl, sigma_ss_nl = self.sigma, self.sigma_dd_nl, self.sigma_sd_nl, self.sigma_ss_nl

        # Compute the growth rate depending on what we have left as free parameters
        if self.fit_growth:
            growth = p["f"]
        else:
            if self.fit_omega_m:
                growth = p["om"]**0.55
            else:
                growth = self.omega_m**0.55

        # Compute the propagator
        nmu = 100
        mu = np.linspace(0.0, 1.0, nmu)
        if self.recon:
            # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
            smoothing_kernel = np.exp(-ks ** 2 * self.recon_smoothing_scale ** 2 / 4.0)

            damping_dd = np.exp(-np.outer(1.0 + (2.0 + growth)*growth*mu**2, ks**2)*sigma_dd_nl)
            damping_sd = np.exp(-np.outer(1.0 + growth*mu**2, ks**2)*sigma_dd_nl)
            damping_ss = np.exp(-np.tile(ks**2, (nmu, 1))*sigma_ss_nl)
            smooth_prefac = np.tile(smoothing_kernel/p["b"], (nmu, 1))
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (nmu, 1))
            kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth/p["b"]*mu**2, 1.0-smoothing_kernel) + bdelta_prefac
            propagator = (kaiser_prefac**2 - bdelta_prefac**2)*damping_dd + 2.0*kaiser_prefac*smooth_prefac*damping_sd + smooth_prefac**2*damping_ss
        else:
            damping = np.exp(-np.outer(1.0 + (2.0 + growth)*growth*mu**2, ks**2)*sigma_nl)
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (nmu, 1))
            kaiser_prefac = 1.0 + np.tile(growth/p["b"]*mu**2, (len(ks), 1)).T + bdelta_prefac
            propagator = (kaiser_prefac**2 - bdelta_prefac**2)*damping

        # Compute the smooth model
        fog = 1.0/(1.0 + np.outer(mu**2, ks**2*p["sigma_s"]**2/2.0))**2
        pk_smooth = p["b"]**2*pk_smooth_lin*fog

        # Integrate over mu
        pk1d = integrate.simps(pk_smooth*(1.0 + pk_ratio*propagator), mu, axis=0)

        # Polynomial shape
        if self.recon:
            shape = p["a1"] * ks**2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d + shape))

        return pk_final

    def get_model(self, data, p):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(data["ks_input"], p)

        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_windowed, mask = self.adjust_model_window_effects(pk_generated)
        return pk_windowed[mask]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model_pre = PowerDing2018(fit_omega_m=True, recon=False)
    model_post = PowerDing2018(fit_omega_m=True, recon=True)

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model_pre.set_data(data)
    model_post.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_s":10.0, "b": 1, "b_delta": 1, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    import timeit
    n = 500

    def test():
        model.get_likelihood(p)

    #print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model_pre.get_model(data, p)
        model_pre.smooth_type = "eh1998"
        pk3 = model_pre.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k', label="Data")
        plt.plot(ks, pk2, '.', c='r', label="hinton2017")
        plt.plot(ks, pk3, '+', c='b', label="eh1998")
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        model_pre.smooth_type = "hinton2017"
        pk_smooth_lin, _ = model_pre.compute_basic_power_spectrum(model_pre.camb.ks, p)
        pk_smooth_interp = splev(data["ks_input"], splrep(model_pre.camb.ks, pk_smooth_lin))
        pk_smooth_lin_windowed, mask = model_pre.adjust_model_window_effects(pk_smooth_interp)
        pk2 = model_pre.get_model(data, p)
        pk3 = model_post.get_model(data, p)
        import matplotlib.pyplot as plt
        plt.plot(ks, pk2/pk_smooth_lin_windowed[mask], '.', c='r', label="pre-recon")
        plt.plot(ks, pk3/pk_smooth_lin_windowed[mask], '+', c='b', label="post-recon")
        plt.xlabel("k")
        plt.ylabel("P(k)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()