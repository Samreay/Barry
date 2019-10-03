import logging
from functools import lru_cache

import numpy as np
from scipy.interpolate import splev, splrep
from scipy import integrate
from barry.models.bao_power import PowerSpectrumFit
from barry.cosmology.camb_generator import Omega_m_z


class PowerDing2018(PowerSpectrumFit):
    """ P(k) model inspired from Ding 2018.

    See https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1021D for details.

    """

    def __init__(self, name="Pk Ding 2018", fix_params=("om", "f"), smooth_type="hinton2017", recon=False, postprocess=None, smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(name=name, fix_params=fix_params, smooth_type=smooth_type, postprocess=postprocess, smooth=smooth, correction=correction)

        self.fit_omega_m = fix_params is None or "om" not in fix_params
        self.fit_growth = fix_params is None or "f" not in fix_params
        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)
        self.smoothing_kernel = None

    @lru_cache(maxsize=32)
    def get_growth(self, om):
        return Omega_m_z(om, self.camb.redshift) ** 0.55

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
            growth = self.get_growth(p["om"])

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
            smooth_prefac = np.tile(self.smoothing_kernel / p["b"], (self.nmu, 1))
            bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
            kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth / p["b"] * self.mu ** 2, 1.0 - self.smoothing_kernel) + bdelta_prefac
            propagator = (
                (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping_dd + 2.0 * kaiser_prefac * smooth_prefac * damping_sd + smooth_prefac ** 2 * damping_ss
            )
        else:
            bdelta_prefac = np.tile(0.5 * p["b_delta"] / p["b"] * ks ** 2, (self.nmu, 1))
            kaiser_prefac = 1.0 + np.tile(growth / p["b"] * self.mu ** 2, (len(ks), 1)).T + bdelta_prefac
            propagator = (kaiser_prefac ** 2 - bdelta_prefac ** 2) * damping

        # Compute the smooth model
        fog = 1.0 / (1.0 + np.outer(self.mu ** 2, ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
        pk_smooth = p["b"] ** 2 * pk_smooth_lin * fog

        # Polynomial shape
        if self.recon:
            shape = p["a1"] * ks ** 2 + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)
        else:
            shape = p["a1"] * ks + p["a2"] + p["a3"] / ks + p["a4"] / (ks * ks) + p["a5"] / (ks ** 3)

        # Integrate over mu
        if smooth:
            pk1d = integrate.simps((pk_smooth + shape) * (1.0 + 0.0 * pk_ratio * propagator), self.mu, axis=0)
        else:
            pk1d = integrate.simps((pk_smooth + shape) * (1.0 + pk_ratio * propagator), self.mu, axis=0)

        pk_final = splev(k / p["alpha"], splrep(ks, pk1d))

        return pk_final


if __name__ == "__main__":

    import sys
    import timeit
    from barry.datasets.dataset_power_spectrum import PowerSpectrum_SDSS_DR12_Z061_NGC

    sys.path.append("../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=False)
    data = dataset.get_data()
    model_pre = PowerDing2018(recon=False)
    model_pre.set_data(data)

    dataset = PowerSpectrum_SDSS_DR12_Z061_NGC(recon=True)
    data = dataset.get_data()
    model_post = PowerDing2018(recon=True)
    model_post.set_data(data)

    p = {"om": 0.3, "alpha": 1.0, "sigma_s": 10.0, "b": 1.6, "b_delta": 1, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    n = 200

    def test_pre():
        model_pre.get_likelihood(p, data[0])

    def test_post():
        model_post.get_likelihood(p, data[0])

    print("Pre-reconstruction likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test_pre, number=n) * 1000 / n))
    print("Post-reconstruction likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test_post, number=n) * 1000 / n))

    if True:
        p, minv = model_pre.optimize()
        print("Pre reconstruction optimisation:")
        print(p)
        print(minv)
        model_pre.plot(p)

        print("Post reconstruction optimisation:")
        p, minv = model_post.optimize()
        print(p)
        print(minv)
        model_post.plot(p)
