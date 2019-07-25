import logging
import numpy as np
from scipy.interpolate import splev, splrep
from barry.framework.models.bao_power import PowerSpectrumFit


class PowerNoda2019(PowerSpectrumFit):

    def __init__(self, fix_params=None, gammaval=None, smooth_type="hinton2017", recon=False, name="Pk Noda 2019", postprocess=None):
        self.recon = recon
        if fix_params is None:
            if recon:
                fix_params = ["om", "f"]
            else:
                fix_params = ["om", "f", "gamma"]

        self.fit_omega_m = fix_params is None or "om" not in fix_params
        self.fit_growth = fix_params is None or "f" not in fix_params
        self.fit_gamma = fix_params is None or "gamma" not in fix_params
        if gammaval is None:
            if self.recon:
                gammaval = 4.0
            else:
                gammaval = 1.0
        self.gammaval = gammaval
        super().__init__(fix_params=fix_params, smooth_type=smooth_type, name=name, postprocess=postprocess)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.omega_m, self.pt_data, self.growth, self.damping = None, None, None, None

    def set_data(self, data):
        super().set_data(data)
        self.omega_m = self.get_default("om")
        if not self.fit_omega_m:
            self.pt_data = self.PT.get_data(om=self.omega_m)
            if not self.fit_growth:
                self.growth = self.omega_m ** 0.55
                self.damping = -np.outer((1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2) * self.pt_data["sigma_dd_rs"] + (self.growth * self.mu ** 2 * (self.mu ** 2 - 1.0)) * self.pt_data["sigma_ss_rs"], self.camb.ks ** 2)
                if not self.fit_gamma:
                    self.damping /= self.gammaval

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("gamma", r"$\gamma_{rec}$", 1.0, 8.0, self.gammaval)  # Describes the sharpening of the BAO post-reconstruction
        self.add_param("A", r"$A$", -10, 30.0, 10)  # Fingers-of-god damping

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

        from scipy import integrate

        # Get the basic power spectrum components
        ks = self.camb.ks
        pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(p["om"])
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
        if self.fit_growth or self.fit_omega_m:
            damping = -np.outer((1.0 + (2.0 + growth) * growth * self.mu ** 2) * pt_data["sigma_dd_rs"] + (
                        growth * self.mu ** 2 * (self.mu ** 2 - 1.0)) * pt_data["sigma_ss_rs"], ks ** 2)
            if self.fit_gamma:
                damping /= p["gamma"]
            else:
                damping /= self.gammaval
        else:
            if self.fit_gamma:
                damping = self.damping/p["gamma"]
            else:
                damping = self.damping
        damping = np.exp(damping)

        # Compute the propagator
        if self.recon:
            # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
            smoothing_kernel = np.exp(-ks ** 2 * self.recon_smoothing_scale ** 2 / 4.0)
            kaiser_prefac = 1.0 + np.outer(growth / p["b"] * self.mu ** 2, 1.0-smoothing_kernel)
        else:
            kaiser_prefac = 1.0 + np.tile(growth / p["b"] * self.mu ** 2, (len(ks), 1)).T
        propagator = kaiser_prefac**2*damping

        # Compute the smooth model
        fog = np.exp(-p["A"]*ks**2)
        pk_smooth = p["b"]**2*pk_smooth_lin*fog

        # Compute the non-linear SPT correction to the smooth power spectrum
        pk_spt = pt_data["I00"] + pt_data["J00"] + 2.0*np.outer(growth/p["b"]*self.mu**2, pt_data["I01"] + pt_data["J01"]) + np.outer((growth/p["b"]*self.mu**2)**2, pt_data["I11"] + pt_data["J11"])

        # Integrate over mu
        if smooth:
            pk1d = integrate.simps(pk_smooth*(1.0 + 0.0 * pk_ratio*propagator + pk_spt), self.mu, axis=0)
        else:
            pk1d = integrate.simps(pk_smooth*(1.0 + pk_ratio*propagator + pk_spt), self.mu, axis=0)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d))

        return pk_final


if __name__ == "__main__":
    import sys
    sys.path.append("../../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    model_pre = PowerNoda2019(recon=False)
    model_post = PowerNoda2019(recon=True, gammaval=4.0)

    from barry.framework.datasets.mock_power import MockTaipanPowerSpectrum
    dataset = MockTaipanPowerSpectrum()
    data = dataset.get_data()
    model_pre.set_data(data)
    model_post.set_data(data)

    p = {"om": 0.3, "alpha": 1.0, "A": 7.0, "b": 1.6, "gamma": 4.0}
    for v in np.linspace(1.0, 20, 20):
        p["A"] = v
        print(v, model_post.get_likelihood(p))

    import timeit
    n = 100

    def test():
        model_post.get_likelihood(p)

    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if False:
        ks = data["ks"]
        pk = data["pk"]
        pk2 = model_pre.get_model(p)
        model_pre.smooth_type = "eh1998"
        pk3 = model_pre.get_model(p)
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
        pk_smooth_lin, _ = model_pre.compute_basic_power_spectrum(p["om"])
        growth = p["om"]**0.55
        pt_data = model_pre.PT.get_data(om=p["om"])
        pk_spt = pt_data["I00"] + pt_data["J00"] + 2.0/3.0*growth/p["b"]*(pt_data["I01"] + pt_data["J01"]) + 1.0/5.0*(growth/p["b"])**2*(pt_data["I11"] + pt_data["J11"])
        pk_smooth_interp = splev(data["ks_input"], splrep(model_pre.camb.ks, pk_smooth_lin*(1.0+pk_spt)))
        pk_smooth_lin_windowed, mask = model_pre.adjust_model_window_effects(pk_smooth_interp)
        pk2 = model_pre.get_model(p)
        pk3 = model_post.get_model(p)
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