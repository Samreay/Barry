import logging
import math
import numpy as np
from scipy import integrate, interpolate, optimize
from scipy.fftpack import dst, idst
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelmin, argrelmax, savgol_filter
from cosmoprimo import Cosmology, PowerSpectrumInterpolator1D, PowerSpectrumBAOFilter
from cosmoprimo.fiducial import DESI


def get_smooth_methods_list():
    fns = ["hinton2017", "wallish2018", "ehpoly", "savgol", "brieden2022", "peakaverage"]
    return fns


def validate_smooth_method(kwargs):
    if "method" in kwargs:
        method = kwargs["method"].lower()
        if method in get_smooth_methods_list():
            return True
        else:
            logging.getLogger("barry").error(f"Smoothing method is {method} and not in list {get_smooth_methods_list()}")
            return False
    else:
        logging.getLogger("barry").error(f"No smoothing method specified in smooth_type: {kwargs}")


def smooth_func(ks, pk, om=0.31, h0=0.676, ob=0.04814, ns=0.97, mnu=0.0, method="hinton2017"):

    # Set up a cosmology
    new_params = {"Omega_m": om, "h": h0, "Omega_b": ob, "n_s": ns, "omega_ncdm": mnu / 93.14}
    cosmo = DESI().clone(engine="camb", **new_params)

    # Set up the power spectrum interpolator for cosmoprimo,
    pk1d = PowerSpectrumInterpolator1D(ks, pk, extrap_kmin=1e-5)

    # Set up the cosmoprimo smooth pk class
    pknow = PowerSpectrumBAOFilter(pk1d, cosmo_fid=cosmo, engine=method).smooth_pk_interpolator()

    return pknow(ks)


if __name__ == "__main__":
    import sys

    sys.path.append("../..")

    import timeit
    import matplotlib.pyplot as plt
    from barry.cosmology.camb_generator import getCambGenerator

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Set up the cosmological model for testing
    c = DESI()
    camb = getCambGenerator(om_resolution=1, h0=c["h"], ob=c["Omega_b"], ns=c["n_s"], mnu=np.sum(c["omega_ncdm"]) * 93.14)
    camb.omch2s = [(c["Omega_m"] - c["Omega_b"]) * c["h"] ** 2 - np.sum(c["omega_ncdm"])]
    ks = camb.ks
    pk_lin = camb.get_data()["pk_lin_0"]

    if False:  # Do timing tests
        n = 30

        def test_hinton():
            smooth_func(ks, pk_lin, method="hinton2017")

        def test_wallish():
            smooth_func(ks, pk_lin, method="wallish2018")

        def test_ehpoly():
            smooth_func(ks, pk_lin, method="ehpoly")

        def test_savgol():
            smooth_func(ks, pk_lin, method="savgol")

        def test_brieden():
            smooth_func(ks, pk_lin, method="brieden2022")

        def test_peakaverage():
            smooth_func(ks, pk_lin, method="peakaverage")

        t_hinton = timeit.timeit(test_hinton, number=n) * 1000 / n
        t_wallish = timeit.timeit(test_wallish, number=n) * 1000 / n
        t_ehpoly = timeit.timeit(test_ehpoly, number=n) * 1000 / n
        t_savgol = timeit.timeit(test_savgol, number=n) * 1000 / n
        t_brieden = timeit.timeit(test_brieden, number=n) * 1000 / n
        t_peakaverage = timeit.timeit(test_peakaverage, number=n) * 1000 / n

        print(f"Hinton smoothing takes on average, {t_hinton:.2f} milliseconds")
        print(f"Wallish smoothing takes on average, {t_wallish:.2f} milliseconds")
        print(f"Eisenstein and Hu smoothing takes on average, {t_ehpoly:.2f} milliseconds")
        print(f"Savitsky-Golay smoothing takes on average, {t_savgol:.2f} milliseconds")
        print(f"Brieden smoothing takes on average, {t_brieden:.2f} milliseconds")
        print(f"Peak-average smoothing takes on average, {t_peakaverage:.2f} milliseconds")

    if True:  # Do plotting comparison
        import matplotlib.pyplot as plt

        labels = [
            r"$\mathrm{Hinton2017}$",
            r"$\mathrm{Wallisch2018}$",
            r"$\mathrm{EH1998}$",
            r"$\mathrm{Savitsky-Golay}$",
            r"$\mathrm{Brieden2022}$",
            r"$\mathrm{Peak-average}$",
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(ks, pk_lin, "-", c="k")
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        for i, smooth_type in enumerate(get_smooth_methods_list()):
            print(i, smooth_type)
            pk_smoothed = smooth_func(ks, pk_lin, method=smooth_type)
            ax1.plot(ks, pk_smoothed, "-", ms=2, label=labels[i])
            ax2.plot(ks, pk_lin / pk_smoothed, "-")
        ax1.legend()
        plt.show()

        fig, (ax1) = plt.subplots(1, 1)
        for i, smooth_type in enumerate(get_smooth_methods_list()):
            print(i, smooth_type)
            pk_smoothed = smooth_func(ks, pk_lin, method=smooth_type)
            ax1.plot(ks, pk_lin / pk_smoothed, "-", label=labels[i])
        ax1.set_xlim(0.0, 0.4)
        ax1.set_xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=14)
        ax1.set_ylabel(r"$P(k)/P_{\mathrm{smooth}}(k)$", fontsize=14)
        ax1.legend()
        plt.show()
