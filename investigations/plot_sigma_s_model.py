import logging
from barry.datasets import CorrelationFunction_DESI_KP4
from barry.models import CorrBeutler2017
from barry.models.model import Correction
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Load some data to set the cosmology etc.
    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        mocktype="abacus_cubicbox",
        redshift_bin=0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )
    data = dataset.get_data()
    err = np.sqrt(np.diag(data[0]["cov"]))
    errs = [row for row in err.reshape((-1, len(data[0]["dist"])))]

    # Generate the wiggle and no-wiggle components of the power spectrum model with two
    # apparently degenerate sets of Sigma values to see how they actually look
    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        fix_params=["om", "alpha", "epsilon", "sigma_nl_par", "sigma_nl_perp", "sigma_s"],
        marg="full",
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=5,
    )
    model.set_default("alpha", 1.0)
    model.set_default("epsilon", 0.0)
    model.set_default("sigma_nl_par", 5.2)
    model.set_default("sigma_nl_perp", 2.2)
    model.set_default("sigma_s", 0.0)
    model.set_data(data)
    sin, xi, poly = model.compute_correlation_function(model.data[0]["dist"], model.get_param_dict(model.get_defaults()))
    _, xismooth, polysmooth = model.compute_correlation_function(
        model.data[0]["dist"], model.get_param_dict(model.get_defaults()), smooth=True
    )
    fit, [new_chi_squared, dof, bband, mods, smooths] = model.sanity_check(dataset)
    # sin, xi, poly = model.compute_correlation_function(model.data[0]["dist"], fit)
    # _, xismooth, polysmooth = model.compute_correlation_function(model.data[0]["dist"], fit, smooth=True)

    model2 = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        fix_params=["om", "alpha", "epsilon", "sigma_nl_par", "sigma_nl_perp", "sigma_s"],
        marg="full",
        poly_poles=dataset.fit_poles,
        correction=Correction.NONE,
        n_poly=5,
    )
    model2.set_default("alpha", 1.0)
    model2.set_default("epsilon", 0.0)
    model2.set_default("sigma_nl_par", 0.0)
    model2.set_default("sigma_nl_perp", 2.2)
    model2.set_default("sigma_s", 4.4)
    model2.set_data(data)
    sin2, xi2, poly2 = model2.compute_correlation_function(model2.data[0]["dist"], model2.get_param_dict(model2.get_defaults()))
    _, xismooth2, polysmooth2 = model2.compute_correlation_function(
        model2.data[0]["dist"], model2.get_param_dict(model2.get_defaults()), smooth=True
    )
    fit2, [new_chi_squared2, dof2, bband2, mods2, smooths2] = model2.sanity_check(dataset)
    # sin2, xi2, poly2 = model2.compute_correlation_function(model2.data[0]["dist"], fit2)

    # First let's plot the models with just the default parameters
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xi[0] - xi2[0]), color="r")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xi[2] - xi2[2]), color="b")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (mods[0] - mods2[0]), color="r", ls="--")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (mods[2] - mods2[2]), color="b", ls="--")
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[0], model.data[0]["dist"] ** 2 * errs[0], color="r", alpha=0.2
    )
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[2], model.data[0]["dist"] ** 2 * errs[2], color="b", alpha=0.2
    )
    plt.show()

    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xismooth[0] - xismooth2[0]), color="r")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xismooth[2] - xismooth2[2]), color="b")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (smooths[0] - smooths2[0]), color="r", ls="--")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (smooths[2] - smooths2[2]), color="b", ls="--")
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[0], model.data[0]["dist"] ** 2 * errs[0], color="r", alpha=0.2
    )
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[2], model.data[0]["dist"] ** 2 * errs[2], color="b", alpha=0.2
    )
    plt.show()

    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xi[0] - xi2[0] - xismooth[0] + xismooth2[0]), color="r")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (xi[2] - xi2[2] - xismooth[2] + xismooth2[2]), color="b")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (mods[0] - mods2[0] - smooths[0] + smooths2[0]), color="r", ls="--")
    plt.plot(model.data[0]["dist"], model.data[0]["dist"] ** 2 * (mods[2] - mods2[2] - smooths[2] + smooths2[2]), color="b", ls="--")
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[0], model.data[0]["dist"] ** 2 * errs[0], color="r", alpha=0.2
    )
    plt.fill_between(
        model.data[0]["dist"], -model.data[0]["dist"] ** 2 * errs[2], model.data[0]["dist"] ** 2 * errs[2], color="b", alpha=0.2
    )
    plt.show()

    """epsilon = 0 if for_corr else p["epsilon"]
    kprime = np.tile(k, (self.nmu, 1)).T if for_corr else np.outer(k / p["alpha"], self.get_kprimefac(epsilon))
    muprime = self.mu if for_corr else self.get_muprime(epsilon)
    if self.dilate_smooth:
        fog = 1.0 / (1.0 + muprime**2 * kprime**2 * p["sigma_s"] ** 2 / 2.0) ** 2
        reconfac = splev(kprime, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
        kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
        pk_smooth = kaiser_prefac**2 * splev(kprime, splrep(ks, pk_smooth_lin)) * fog
    else:
        ktile = np.tile(k, (self.nmu, 1)).T
        fog = 1.0 / (1.0 + muprime**2 * ktile**2 * p["sigma_s"] ** 2 / 2.0) ** 2
        reconfac = splev(ktile, splrep(self.camb.ks, self.camb.smoothing_kernel)) if self.recon_type.lower() == "iso" else 0.0
        kaiser_prefac = 1.0 + p["beta"] * muprime**2 * (1.0 - reconfac)
        pk_smooth = kaiser_prefac**2 * splev(ktile, splrep(ks, pk_smooth_lin))

    if not for_corr:
        pk_smooth *= p["b{0}"]

    # Volume factor
    pk_smooth /= p["alpha"] ** 3

    # Compute the propagator
    if smooth:
        pk2d = pk_smooth * fog
    else:
        C = np.exp(-0.5 * kprime**2 * (muprime**2 * p["sigma_nl_par"] ** 2 + (1.0 - muprime**2) * p["sigma_nl_perp"] ** 2))
        pk2d = pk_smooth * (fog + splev(kprime, splrep(ks, pk_ratio)) * C)

    pk0, pk2, pk4 = self.integrate_mu(pk2d)"""
