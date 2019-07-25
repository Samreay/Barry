from functools import lru_cache

from scipy.interpolate import splev, splrep

from barry.framework.cosmology.PT_generator import PTGenerator
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method
from barry.framework.model import Model
import numpy as np


# TODO: make h0 and omega_m more easily changable if we are not fitting them
class PowerSpectrumFit(Model):
    def __init__(self, smooth_type="hinton2017", name="Pk Basic", postprocess=None, fix_params=['om'], smooth=False, correction=None):
        super().__init__(name, postprocess=postprocess, correction=correction)
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.smooth = smooth
        self.camb = None
        self.PT = None
        self.recon_smoothing_scale = None
        self.cosmology = None

    def set_data(self, data):
        super().set_data(data)
        c = data[0]["cosmology"]
        if self.cosmology != c:
            self.recon_smoothing_scale = c["reconsmoothscale"]
            self.camb = CambGenerator(h0=c["h0"], ob=c["ob"], redshift=c["z"], ns=c["ns"])
            self.PT = PTGenerator(self.camb, smooth_type=self.smooth_type, recon_smoothing_scale=self.recon_smoothing_scale)
            self.set_default("om", c["om"])

    def declare_parameters(self):
        # Define parameters
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch
        self.add_param("b", r"$b$", 0.1, 12.5, 1.73)  # bias

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed, linear power spectrum and the wiggle ratio

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        array
            pk_smooth - The power spectrum smoothed out
        array
            pk_ratio_dewiggled - the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        r_s, pk_lin = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=om, h0=self.camb.h0)  # Get the smoothed power spectrum
        pk_ratio = (pk_lin / pk_smooth_lin - 1.0)  # Get the ratio
        return pk_smooth_lin, pk_ratio

    def compute_power_spectrum(self, k, p, smooth=False):
        """ Returns the wiggle ratio interpolated at some k/alpha values. Useful if we only want alpha to modify
            the BAO feature and not the smooth component.

        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers to compute.
        p : dict
            dictionary of parameter names to their values

        Returns
        -------
        array
            pk_final - the ratio (pk_lin / pk_smooth - 1.0),  interpolated to k/alpha.

        """
        ks = self.camb.ks
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum(p["om"])
        if smooth:
            pk_final = splev(k / p["alpha"], splrep(ks, pk_smooth))
        else:
            pk_final = splev(k / p["alpha"], splrep(ks, pk_smooth * (1 + pk_ratio_dewiggled)))
        return pk_final

    def adjust_model_window_effects(self, pk_generated, data):
        p0 = np.sum(data["w_scale"] * pk_generated)
        integral_constraint = data["w_pk"] * p0

        pk_convolved = np.atleast_2d(pk_generated) @ data["w_transform"]
        pk_normalised = (pk_convolved - integral_constraint).flatten()
        # Get the subsection of our model which corresponds to the data k values
        return pk_normalised, data["w_mask"]

    def get_likelihood(self, p, d):
        pk_model = self.get_model(p, d, smooth=self.smooth)

        # Compute the chi2
        diff = (d["pk"] - pk_model)
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())
        return self.get_chi2_likelihood(diff, d["icov"], num_mocks=num_mocks, num_params=num_params)

    def get_model(self, p, d, smooth=False):
        # Get the generic pk model
        pk_generated = self.compute_power_spectrum(d["ks_input"], p, smooth=smooth)
        # Morph it into a model representative of our survey and its selection/window/binning effects
        pk_model, mask = self.adjust_model_window_effects(pk_generated, d)

        if self.postprocess is not None:
            pk_model = self.postprocess(ks=d["ks_output"], pk=pk_model, mask=mask)
        else:
            pk_model = pk_model[mask]
        return pk_model

    def plot(self, params, smooth_params=None):
        import matplotlib.pyplot as plt

        ks = self.data[0]["ks"]
        pk = self.data[0]["pk"]
        err = np.sqrt(np.diag(self.data[0]["cov"]))
        pk2 = self.get_model(params, self.data[0])

        if smooth_params is not None:
            smooth = self.get_model(smooth_params, self.data[0], smooth=True)
        else:
            smooth = self.get_model(params, self.data[0], smooth=True)

        def adj(data, err=False):
            if self.postprocess is None:
                return data / smooth
            else:
                if err:
                    return data / pk
                else:
                    return data / pk

        fig, axes = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        axes[0].errorbar(ks, ks*pk, yerr=ks*err, fmt="o", c='k', ms=4, label=self.data[0]["name"])
        axes[1].errorbar(ks, adj(pk), yerr=adj(err, err=True), fmt="o", c='k', ms=4, label=self.data[0]["name"])

        # pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(params["om"])
        # axes[1].plot(ks * params["alpha"], 1 + splev(ks, splrep(self.camb.ks, pk_ratio)), label="pkratio", c="r", ls="--")

        axes[0].plot(ks, ks*pk2, label=self.get_name())
        axes[1].plot(ks, adj(pk2), label=self.get_name())

        string = f"Likelihood: {self.get_likelihood(params, self.data[0]):0.2f}\n"
        string += "\n".join([f"{self.param_dict[l].label}={v:0.3f}" for l, v in params.items()])
        va = "center" if self.postprocess is None else "top"
        ypos = 0.5 if self.postprocess is None else 0.98
        axes[0].annotate(string, (0.98, ypos), xycoords="axes fraction", horizontalalignment="right", verticalalignment=va)
        axes[1].legend()
        axes[1].set_xlabel("k")
        if self.postprocess is None:
            axes[1].set_ylabel("P(k) / P_{smooth}(k)")
        else:
            axes[1].set_ylabel("P(k) / data")
        axes[0].set_ylabel("k * P(k)")
        plt.show()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    model = PowerSpectrumFit()

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(step_size=2)
    data = dataset.get_data()
    model.set_data(data)
    p = {"om": 0.3, "alpha": 1.0, "sigma_nl": 5, "b": 0, "a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0}

    import timeit
    n = 500

    def test():
        model.get_posterior(p)

    print("Likelihood takes on average, %.2f milliseconds" % (timeit.timeit(test, number=n) * 1000 / n))

    # if True:
    #     ks = data["ks"]
    #     pk = data["pk"]
    #     pk2 = model.get_model(data, p)
    #     model.smooth_type = "eh1998"
    #     pk3 = model.get_model(data, p)
    #     import matplotlib.pyplot as plt
    #     plt.errorbar(ks, pk, yerr=np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    #     plt.plot(ks, pk2, '.', c='r')
    #     plt.plot(ks, pk3, '+', c='b')
    #     plt.xlabel("k")
    #     plt.ylabel("P(k)")
    #     plt.show()

