from functools import lru_cache

from scipy.interpolate import splev, splrep

from barry.cosmology.PT_generator import getCambGeneratorAndPT
from barry.cosmology.camb_generator import Omega_m_z
from barry.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method
from barry.models.model import Model
import numpy as np


class PowerSpectrumFit(Model):
    """ Generic power spectrum model """

    def __init__(self, name="Pk Basic", smooth_type="hinton2017", fix_params=("om"), postprocess=None, smooth=False, correction=None):
        """ Generic power spectrum function model

        Parameters
        ----------
        name : str, optional
            Name of the model
        smooth_type : str, optional
            The sort of smoothing to use. Either 'hinton2017' or 'eh1998'
        fix_params : list[str], optional
            Parameter names to fix to their defaults. Defaults to just `[om]`.
        postprocess : `Postprocess` object
            The class to postprocess model predictions. Defaults to none.
        smooth : bool, optional
            Whether to generate a smooth model without the BAO feature. Defaults to `false`.
        correction : `Correction` enum.
            Defaults to `Correction.SELLENTIN
        """
        super().__init__(name, postprocess=postprocess, correction=correction)
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)

        self.declare_parameters()
        self.set_fix_params(fix_params)

        # Set up data structures for model fitting
        self.smooth = smooth

    def declare_parameters(self):
        """ Defines model parameters, their bounds and default value. """
        self.add_param("om", r"$\Omega_m$", 0.1, 0.5, 0.31)  # Cosmology
        self.add_param("alpha", r"$\alpha$", 0.8, 1.2, 1.0)  # Stretch
        self.add_param("b", r"$b$", 0.1, 12.5, 1.73)  # bias

    @lru_cache(maxsize=1024)
    def compute_basic_power_spectrum(self, om):
        """ Computes the smoothed linear power spectrum and the wiggle ratio

        Parameters
        ----------
        om : float
            The Omega_m value to generate a power spectrum for

        Returns
        -------
        pk_smooth_lin : np.ndarray
            The power spectrum smoothed out
        pk_ratio : np.ndarray
            the ratio pk_lin / pk_smooth, transitioned using sigma_nl

        """
        # Get base linear power spectrum from camb
        r_s, pk_lin, _ = self.camb.get_data(om=om, h0=self.camb.h0)
        pk_smooth_lin = smooth(self.camb.ks, pk_lin, method=self.smooth_type, om=om, h0=self.camb.h0)  # Get the smoothed power spectrum
        pk_ratio = pk_lin / pk_smooth_lin - 1.0  # Get the ratio
        return pk_smooth_lin, pk_ratio

    def compute_power_spectrum(self, p, smooth=False, shape=True):
        """ Returns the wiggle ratio interpolated at some k/alpha values. Useful if we only want alpha to modify
            the BAO feature and not the smooth component.

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
        pk_1d : np.ndarray
            the ratio (pk_lin / pk_smooth - 1.0),  NOT interpolated to k/alpha.

        """
        ks = self.camb.ks
        pk_smooth, pk_ratio_dewiggled = self.compute_basic_power_spectrum(p["om"])
        if smooth:
            pk_1d = pk_smooth
        else:
            pk_1d = pk_smooth * (1 + pk_ratio_dewiggled)
        return ks, pk_1d

    def adjust_model_window_effects(self, pk_generated, data):
        """ Take the window effects into account.

        Parameters
        ----------
        pk_generated : np.ndarray
            The p(k) values generated at the window function input ks
        data : dict
            The data dictionary containing the window scale `w_scale`,
            transformation matrix `w_transform`, integral constraint `w_pk`
            and mask `w_mask`

        Returns
        -------
        pk_normalised : np.ndarray
            The transformed, corrected power spectrum
        mask : np.ndarray
            A boolean mask used for selecting the final data out of pk_normalised.
            Mask is not applied in this function because for the BAO extractor
            post processing we want to take the powers outside the mask into account
            and *then* mask.
        """
        p0 = np.sum(data["w_scale"] * pk_generated)
        integral_constraint = data["w_pk"] * p0

        pk_convolved = np.atleast_2d(pk_generated) @ data["w_transform"]
        pk_normalised = (pk_convolved - integral_constraint).flatten()
        # Get the subsection of our model which corresponds to the data k values
        return pk_normalised, data["w_mask"]

    def get_likelihood(self, p, d):
        """ Uses the stated likelihood correction and `get_model` to compute the likelihood

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.

        Returns
        -------
        log_likelihood : float
            The corrected log likelihood
        """
        pk_model = self.get_model(p, d, smooth=self.smooth)

        # Compute the chi2
        diff = d["pk"] - pk_model
        num_mocks = d["num_mocks"]
        num_params = len(self.get_active_params())
        return self.get_chi2_likelihood(diff, d["icov"], num_mocks=num_mocks, num_params=num_params)

    def get_model(self, p, d, smooth=False):
        """ Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        data : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.
        smooth : bool, optional
            Whether to only generate a smooth model without the BAO feature

        Returns
        -------
        pk_model : np.ndarray
            The p(k) predictions given p and data, k values correspond to d['ks_output']

        """
        # Get the generic pk model
        ks, pk1d = self.compute_power_spectrum(p, smooth=smooth)

        pk_generated = splev(d["ks_input"] / p["alpha"], splrep(ks, pk1d))
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

        axes[0].errorbar(ks, ks * pk, yerr=ks * err, fmt="o", c="k", ms=4, label=self.data[0]["name"])
        axes[1].errorbar(ks, adj(pk), yerr=adj(err, err=True), fmt="o", c="k", ms=4, label=self.data[0]["name"])

        # pk_smooth_lin, pk_ratio = self.compute_basic_power_spectrum(params["om"])
        # axes[1].plot(ks * params["alpha"], 1 + splev(ks, splrep(self.camb.ks, pk_ratio)), label="pkratio", c="r", ls="--")

        axes[0].plot(ks, ks * pk2, label=self.get_name())
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

    from barry.datasets.mock_power import MockPowerSpectrum

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
