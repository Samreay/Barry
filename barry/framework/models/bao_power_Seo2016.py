import logging
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.integrate import simps
import sys
sys.path.append("../../..")
from barry.framework.models.bao_power import PowerSpectrumFit
from barry.framework.cosmology.PT_generator import PTGenerator


class PowerSeo2016(PowerSpectrumFit):

    def __init__(self, fit_omega_m=False, fit_growth=False, smooth_type="hinton2017", recon=False, recon_smoothing_scale=21.21, name="Pk Seo 2016", postprocess=None):
        self.recon = recon
        self.recon_smoothing_scale = recon_smoothing_scale
        self.fit_growth = fit_growth
        super().__init__(fit_omega_m=fit_omega_m, smooth_type=smooth_type, name=name, postprocess=postprocess)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

        self.PT = PTGenerator(self.camb, smooth_type=self.smooth_type, recon_smoothing_scale=self.recon_smoothing_scale)
        if not self.fit_omega_m:
            self.pt_data = self.PT.get_data(om=self.omega_m)
            if not self.fit_growth:
                self.growth = self.omega_m ** 0.55
                if self.recon:
                    self.damping_dd = np.exp(-np.outer(1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2, self.camb.ks ** 2) * self.pt_data["sigma_dd"] / 2.0)
                    self.damping_ss = np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.pt_data["sigma_ss"] / 2.0)
                else:
                    self.damping = np.exp(-np.outer(1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2, self.camb.ks ** 2) * self.pt_data["sigma"] / 2.0)

        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        if self.recon:
            self.smoothing_kernel = np.exp(-self.camb.ks ** 2 * self.recon_smoothing_scale ** 2 / 4.0)

    def declare_parameters(self):
        super().declare_parameters()
        if self.fit_growth:
            self.add_param("f", r"$f$", 0.01, 1.0)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0)  # Fingers-of-god damping
        self.add_param("a1", r"$a_1$", -50000.0, 50000.0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -50000.0, 50000.0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -50000.0, 50000.0)  # Polynomial marginalisation 3
        self.add_param("a4", r"$a_4$", -1000.0, 1000.0)  # Polynomial marginalisation 4
        self.add_param("a5", r"$a_5$", -10.0, 10.0)  # Polynomial marginalisation 5

    def compute_power_spectrum(self, k, p):
        """ Computes the power spectrum model using the LPT based propagators from Seo et. al., 2016 at k/alpha
        
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
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p)
        if self.fit_omega_m:
            pt_data = self.PT.get_data(om=p["om"])
        else:
            pt_data = self.pt_data

        # Compute the growth rate depending on what we have left as free parameters
        if self.fit_growth:
            growth = p["f"]
        else:
            if self.fit_omega_m:
                growth = p["om"]**0.55
            else:
                growth = self.growth

        # Compute the BAO damping
        if self.recon:
            if self.fit_growth or self.fit_omega_m:
                damping_dd = np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, ks ** 2) * pt_data["sigma_dd"] / 2.0)
                damping_ss = np.exp(-np.tile(ks ** 2, (self.nmu, 1)) * pt_data["sigma_ss"] / 2.0)
            else:
                damping_dd, damping_ss = self.damping_dd, self.damping_ss
        else:
            if self.fit_growth or self.fit_omega_m:
                damping = np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, ks ** 2) * pt_data["sigma"] / 2.0)
            else:
                damping = self.damping

        # Compute the propagator
        if self.recon:
            kaiser = 1.0 + np.outer(growth/p["b"]*self.mu**2, 1.0-self.smoothing_kernel)
            smooth_prefac = np.tile(self.smoothing_kernel/p["b"], (self.nmu, 1))
            propagator = (kaiser*damping_dd + smooth_prefac*(damping_ss-damping_dd))**2
        else:
            prefac_k = 1.0 + np.tile(3.0/7.0*(pt_data["R1"]*(1.0-4.0/(9.0*p["b"]))+pt_data["R2"]), (self.nmu, 1))
            prefac_mu = np.outer(self.mu**2, growth/p["b"] + 3.0/7.0*growth*pt_data["R1"]*(2.0-1.0/(3.0*p["b"])) + 6.0/7.0*growth*pt_data["R2"])
            propagator = ((prefac_k + prefac_mu)*damping)**2

        # Compute the smooth model
        fog = 1.0/(1.0 + np.outer(self.mu**2, ks**2*p["sigma_s"]**2/2.0))**2
        pk_smooth = p["b"]**2*pk_smooth_lin*fog

        # Integrate over mu
        pk1d = simps(pk_smooth*(1.0 + pk_ratio*propagator), self.mu, axis=0)

        # Polynomial shape
        if self.recon:
            shape = p["a1"] * ks**2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d + shape))

        return pk_final


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model_pre = PowerSeo2016(fit_omega_m=True, recon=False)
    model_post = PowerSeo2016(fit_omega_m=True, recon=True)

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model_pre.set_data(data)
    model_post.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_s":10.0, "b": 1.6, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    import timeit
    n = 500

    def test():
        model_post.get_likelihood(p)

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
        pk_smooth_lin, _ = model_pre.compute_basic_power_spectrum(p)
        pk_smooth_interp = splev(data["ks_input"], splrep(model_pre.camb.ks, pk_smooth_lin))
        pk_smooth_lin_windowed, mask = model_pre.adjust_model_window_effects(pk_smooth_interp)
        pk2 = model_pre.get_model(data, p)
        pk3 = model_post.get_model(data, p)
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