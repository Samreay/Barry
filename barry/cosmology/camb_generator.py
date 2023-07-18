from functools import lru_cache

import numpy as np
import inspect
import os
import logging


# TODO: Add options for mnu, h0 default, omega_b, etc


@lru_cache(maxsize=32)
def getCambGenerator(
    redshift=0.51, om_resolution=101, h0_resolution=1, h0=0.676, ob=0.04814, ns=0.97, mnu=0.0, recon_smoothing_scale=21.21
):
    return CambGenerator(
        redshift=redshift,
        om_resolution=om_resolution,
        h0_resolution=h0_resolution,
        h0=h0,
        ob=ob,
        ns=ns,
        mnu=mnu,
        recon_smoothing_scale=recon_smoothing_scale,
    )


def Omega_m_z(omega_m, z):
    """
    Computes the matter density at redshift based on the present day value.

    Assumes Flat LCDM cosmology, which is fine given this is also assumed in CambGenerator. Possible improvement
    could be to tabulate this using the CambGenerator so that it would be self consistent for non-LCDM cosmologies.

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the matter density at
    :return: the matter density at redshift z
    """
    return omega_m * (1.0 + z) ** 3 / E_z(omega_m, z) ** 2


def E_z(omega_m, z):
    """
    Compute the E-function; the ratio of the Hubble parameter at redshift z to the Hubble-Lemaitre constant.

    Assumes Flat LCDM cosmology, which is fine given this is also assumed in CambGenerator. Would not be necessary if
    we tabulated Omega_m_z using the CambGenerator.

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the E-function at
    :return: The E-function at redshift z given the matter density
    """
    return np.sqrt((1.0 + z) ** 3 * omega_m + (1.0 - omega_m))


class CambGenerator(object):
    """An object to generate power spectra using camb and save them to file.

    Useful because computing them in a likelihood step is insanely slow.
    """

    def __init__(
        self, redshift=0.61, om_resolution=101, h0_resolution=1, h0=0.676, ob=0.04814, ns=0.97, mnu=0.0, recon_smoothing_scale=21.21
    ):
        """
        Precomputes CAMB for efficiency. Access ks via self.ks, and use get_data for an array
        of both the linear and non-linear power spectrum
        """
        self.logger = logging.getLogger("barry")
        self.om_resolution = om_resolution
        self.h0_resolution = h0_resolution
        self.h0 = h0
        self.redshift = redshift
        self.singleval = True if om_resolution == 1 and h0_resolution == 1 else False

        self.data_dir = os.path.normpath(os.path.dirname(inspect.stack()[0][1]) + "/../generated/")
        hh = int(h0 * 10000)
        self.filename_unique = f"{int(self.redshift * 1000)}_{self.om_resolution}_{self.h0_resolution}_{hh}_{int(ob * 10000)}_{int(ns * 1000)}_{int(mnu * 10000)}"
        self.filename = self.data_dir + f"/camb_{self.filename_unique}.npy"

        self.k_min = 1e-4
        self.k_max = 100
        self.k_num = 2000
        self.ks = np.logspace(np.log(self.k_min), np.log(self.k_max), self.k_num, base=np.e)
        self.recon_smoothing_scale = recon_smoothing_scale
        self.smoothing_kernel = np.exp(-self.ks**2 * self.recon_smoothing_scale**2 / 2.0)

        self.omch2s = np.linspace(0.05, 0.3, self.om_resolution)
        self.omega_b = ob
        self.ns = ns
        self.mnu = mnu
        if h0_resolution == 1:
            self.h0s = [h0]
        else:
            self.h0s = np.linspace(0.6, 0.8, self.h0_resolution)

        self.data = None
        self.logger.info(f"Creating CAMB data with {self.om_resolution} x {self.h0_resolution}")

    def load_data(self, can_generate=False):
        if not os.path.exists(self.filename):
            if not can_generate:
                msg = "Data does not exist and this isn't the time to generate it!"
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.data = self._generate_data()
        else:
            self.data = np.load(self.filename)
            self.logger.info("Loading existing CAMB data")

    @lru_cache(maxsize=512)
    def get_data(self, om=0.31, h0=None):
        """Returns the sound horizon, the linear power spectrum, and the halofit power spectrum at self.redshift"""
        if h0 is None:
            h0 = self.h0
        if self.data is None:
            # If we are not interested in varying om, we can run CAMB this once to avoid precomputing
            if self.singleval:
                self.logger.info(f"Running CAMB")
                self.data = self._generate_data(savedata=False)[0, 0]
            else:
                self.load_data()
        if self.singleval:
            data = self.data
        else:
            omch2 = (om - self.omega_b) * h0 * h0
            data = self._interpolate(omch2, h0)
        return {
            "om": om,
            "h0": h0,
            "r_s": data[0],
            "ks": self.ks,
            "pk_lin": data[1 : 1 + self.k_num],
            "pk_nl_0": data[1 + 1 * self.k_num : 1 + 2 * self.k_num],
            "pk_nl_z": data[1 + 2 * self.k_num :],
        }

    def _generate_data(self, savedata=True):
        self.logger.info(f"Generating CAMB data with {self.om_resolution} x {self.h0_resolution}")
        os.makedirs(self.data_dir, exist_ok=True)
        import camb

        pars = camb.CAMBparams()
        pars.set_dark_energy(w=-1.0, dark_energy_model="fluid")
        pars.InitPower.set_params(As=2.083e-9, ns=self.ns)
        pars.set_matter_power(redshifts=[self.redshift, 0.0], kmax=self.k_max)
        self.logger.info("Configured CAMB power and dark energy")

        data = np.zeros((self.om_resolution, self.h0_resolution, 1 + 3 * self.k_num))
        for i, omch2 in enumerate(self.omch2s):
            for j, h0 in enumerate(self.h0s):
                self.logger.info("Generating %d:%d  %0.4f  %0.4f" % (i, j, omch2, h0))
                pars.set_cosmology(
                    H0=h0 * 100,
                    omch2=omch2,
                    mnu=self.mnu,
                    ombh2=self.omega_b * h0 * h0,
                    omk=0.0,
                    tau=0.066,
                    neutrino_hierarchy="degenerate",
                    num_massive_neutrinos=1,
                )
                pars.NonLinear = camb.model.NonLinear_none
                results = camb.get_results(pars)
                params = results.get_derived_params()
                rdrag = params["rdrag"] * h0
                kh, z, pk_lin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                pars.NonLinear = camb.model.NonLinear_pk
                results.calc_power_spectra(pars)
                kh, z, pk_nonlin = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                data[i, j, 0] = rdrag
                data[i, j, 1 : 1 + self.k_num] = pk_lin[1, :]
                data[i, j, 1 + self.k_num :] = pk_nonlin.flatten()
        if savedata:
            self.logger.info(f"Saving to {self.filename}")
            np.save(self.filename, data)
        return data

    def interpolate(self, om, h0, data=None):
        omch2 = (om - self.omega_b) * h0 * h0
        return self._interpolate(omch2, h0, data=data)

    def _interpolate(self, omch2, h0, data=None):
        """Performs bilinear interpolation on the entire pk array"""
        omch2_index = 1.0 * (self.om_resolution - 1) * (omch2 - self.omch2s[0]) / (self.omch2s[-1] - self.omch2s[0])

        # If omch2 == self.omch2s[-1] we can get an index out of bounds later due to rounding errors, so we
        # manually set the edge cases
        if omch2 >= self.omch2s[-1]:
            omch2_index = self.om_resolution - 1 - 1.0e-6

        if self.h0_resolution == 1:
            h0_index = 0
        else:
            h0_index = 1.0 * (self.h0_resolution - 1) * (h0 - self.h0s[0]) / (self.h0s[-1] - self.h0s[0])

            # If h0 == self.h0s[-1] we can get an index out of bounds later due to rounding errors, so we
            # manually set the edge cases
            if h0 == self.h0s[-1]:
                h0_index = self.h0_resolution - 1 - 1.0e-6

        x = omch2_index - np.floor(omch2_index)
        y = h0_index - np.floor(h0_index)

        if data is None:
            data = self.data
        v1 = data[int(np.floor(omch2_index)), int(np.floor(h0_index))]  # 00
        v2 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index))]  # 01

        if self.h0_resolution == 1:
            final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y)
        else:
            v3 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index))]  # 10
            v4 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index))]  # 11
            final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y) + v3 * y * (1 - x) + v4 * x * y
        return final


def test_rand_h0const():
    g = CambGenerator()
    g.load_data()

    def fn():
        g.get_data(np.random.uniform(0.1, 0.2))

    return fn


if __name__ == "__main__":

    import timeit
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    c = getCambGenerator()

    n = 10000
    print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))

    plt.plot(c.ks, c.get_data(0.2)["pk_lin"], color="b", linestyle="-", label=r"$\mathrm{Linear}\,\Omega_{m}=0.2$")
    plt.plot(c.ks, c.get_data(0.3)["pk_lin"], color="r", linestyle="-", label=r"$\mathrm{Linear}\,\Omega_{m}=0.3$")
    plt.plot(c.ks, c.get_data(0.2)["pk_nl_z"], color="b", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2$")
    plt.plot(c.ks, c.get_data(0.3)["pk_nl_z"], color="r", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
