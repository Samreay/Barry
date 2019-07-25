import logging
import numpy as np
from scipy import integrate
from barry.framework.models.bao_correlation import CorrelationPolynomial


class CorrDing2018(CorrelationPolynomial):

    def __init__(self, recon=False, smooth_type="hinton2017", name="Corr Ding 2018", fix_params=['om', 'f'], smooth=False, correction=None):
        self.recon = recon
        self.recon_smoothing_scale = None
        super().__init__(smooth_type, name, fix_params, smooth, correction=correction)

        self.nmu = 100
        self.mu = np.linspace(0.0, 1.0, self.nmu)

        self.fit_omega_m = fix_params is None or "om" not in fix_params
        self.fit_growth = fix_params is None or "f" not in fix_params

        self.omega_m, self.pt_data, self.damping_dd, self.damping_sd = None, None, None, None
        self.damping_ss, self.damping, self.growth = None, None, None
        self.smoothing_kernel = None

    def set_data(self, data):
        super().set_data(data)
        self.omega_m = self.get_default("om")
        if not self.fit_omega_m:
            self.pt_data = self.PT.get_data(om=self.omega_m)
            if not self.fit_growth:
                self.growth = self.omega_m ** 0.55
                if self.recon:
                    self.damping_dd = np.exp(-np.outer(1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2, self.camb.ks ** 2) * self.pt_data["sigma_dd_nl"])
                    self.damping_sd = np.exp(-np.outer(1.0 + self.growth * self.mu ** 2, self.camb.ks ** 2) * self.pt_data["sigma_dd_nl"])
                    self.damping_ss = np.exp(-np.tile(self.camb.ks ** 2, (self.nmu, 1)) * self.pt_data["sigma_ss_nl"])
                else:
                    self.damping = np.exp(-np.outer(1.0 + (2.0 + self.growth) * self.growth * self.mu ** 2, self.camb.ks ** 2) * self.pt_data["sigma_nl"])

        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        if self.recon:
            self.smoothing_kernel = np.exp(-self.camb.ks ** 2 * self.recon_smoothing_scale ** 2 / 4.0)

    def declare_parameters(self):
        # Define parameters
        super().declare_parameters()
        self.add_param("f", r"$f$", 0.01, 1.0, 0.5)  # Growth rate of structure
        self.add_param("sigma_s", r"$\Sigma_s$", 0.01, 10.0, 5.0)  # Fingers-of-god damping
        self.add_param("b_delta", r"$b_{\delta}$", 0.01, 10.0, 5.0)  # Non-linear galaxy bias
        self.add_param("a1", r"$a_1$", -100, 100, 0)  # Polynomial marginalisation 1
        self.add_param("a2", r"$a_2$", -2, 2, 0)  # Polynomial marginalisation 2
        self.add_param("a3", r"$a_3$", -0.2, 0.2, 0)  # Polynomial marginalisation 3

    def compute_correlation_function(self, d, p, smooth=False):
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
                growth = p["om"] ** 0.55
            else:
                growth = self.growth

        # Compute the BAO damping
        if self.recon:
            if self.fit_growth or self.fit_omega_m:
                damping_dd = np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, ks ** 2) * pt_data["sigma_dd_nl"])
                damping_sd = np.exp(-np.outer(1.0 + growth * self.mu ** 2, ks ** 2) * pt_data["sigma_dd_nl"])
                damping_ss = np.exp(-np.tile(ks ** 2, (self.nmu, 1)) * pt_data["sigma_ss_nl"])
            else:
                damping_dd, damping_sd, damping_ss = self.damping_dd, self.damping_sd, self.damping_ss
        else:
            if self.fit_growth or self.fit_omega_m:
                damping = np.exp(-np.outer(1.0 + (2.0 + growth) * growth * self.mu ** 2, ks ** 2) * pt_data["sigma_nl"])
            else:
                damping = self.damping

        # Compute the propagator
        if self.recon:
            smooth_prefac = np.tile(self.smoothing_kernel/p["b"], (self.nmu, 1))
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (self.nmu, 1))
            kaiser_prefac = 1.0 - smooth_prefac + np.outer(growth/p["b"]*self.mu**2, 1.0-self.smoothing_kernel) + bdelta_prefac
            propagator = (kaiser_prefac**2 - bdelta_prefac**2)*damping_dd + 2.0*kaiser_prefac*smooth_prefac*damping_sd + smooth_prefac**2*damping_ss
        else:
            bdelta_prefac = np.tile(0.5*p["b_delta"]/p["b"]*ks**2, (self.nmu, 1))
            kaiser_prefac = 1.0 + np.tile(growth/p["b"]*self.mu**2, (len(ks), 1)).T + bdelta_prefac
            propagator = (kaiser_prefac**2 - bdelta_prefac**2)*damping

        # Compute the smooth model
        fog = 1.0 / (1.0 + np.outer(self.mu ** 2, ks ** 2 * p["sigma_s"] ** 2 / 2.0)) ** 2
        pk_smooth = p["b"] ** 2 * pk_smooth_lin * fog

        # Integrate over mu
        if smooth:
            pk1d = integrate.simps(pk_smooth, self.mu, axis=0)
        else:
            pk1d = integrate.simps(pk_smooth * (1.0 + pk_ratio * propagator), self.mu, axis=0)

        # Convert to correlation function and take alpha into account
        xi = self.pk2xi.pk2xi(ks, pk1d, d * p["alpha"])

        # Polynomial shape
        shape = p["a1"] / (d ** 2) + p["a2"] / d + p["a3"]

        # Add poly shape to xi model, include bias correction
        model = xi + shape
        return model


if __name__ == "__main__":
    import sys
    sys.path.append("../../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    bao = CorrDing2018()

    from barry.framework.datasets.mock_correlation import MockSDSSdr7CorrelationFunction
    dataset = MockSDSSdr7CorrelationFunction()
    data = dataset.get_data()
    bao.set_data(data)

    import timeit
    n = 500
    p = {"om":0.3, "alpha":1.0, "f": 1.0, "sigma_s": 5.0, "b_delta": 5.0, "b": 1.0, "a1": 0, "a2": 0, "a3": 0}

    def test():
        bao.get_likelihood(p)
    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    if True:
        ss = data["dist"]
        xi0 = data["xi0"]
        xi = bao.compute_correlation_function(ss, p)
        print(xi0)
        print(xi)
        import matplotlib.pyplot as plt
        plt.errorbar(ss, ss * ss * xi, yerr=ss * ss * np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
        plt.plot(ss, ss * ss * xi0, c='r')
        plt.show()
