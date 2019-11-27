from functools import lru_cache

import numpy as np
import inspect
import os
import logging
from scipy import integrate, special, interpolate
import pickle
import sys


sys.path.append("../..")
from barry.config import get_config, is_local
from barry.cosmology.camb_generator import getCambGenerator, Omega_m_z, E_z
from barry.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method


@lru_cache(maxsize=32)
def getCambGeneratorAndPT(redshift=0.51, om_resolution=101, h0_resolution=1, h0=0.676, ob=0.04814, ns=0.97, smooth_type="hinton2017", recon_smoothing_scale=15):
    c = getCambGenerator(redshift=redshift, om_resolution=om_resolution, h0_resolution=h0_resolution, h0=h0, ob=ob, ns=ns)
    pt = PTGenerator(c, smooth_type=smooth_type, recon_smoothing_scale=recon_smoothing_scale)
    return c, pt


def Growth_factor_Linder(omega_m, z, gamma=0.55):
    """
    Computes the unnormalised growth factor at redshift z given the present day value of omega_m. Uses the approximation
    from Linder2005 with fixed gamma

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the matter density at
    :param gamma: the growth index. Default of 0.55 corresponding to LCDM.
    :return: the unnormalised growth factor at redshift z.
    """
    avals = np.logspace(-4.0, np.log10(1.0 / (1.0 + z)), 10000)
    f = Omega_m_z(omega_m, 1.0 / avals - 1.0) ** gamma
    integ = integrate.simps((f - 1.0) / avals, avals, axis=0)
    return np.exp(integ) / (1.0 + z)


def Growth_factor_Heath(omega_m, z):
    """
    Computes the unnormalised growth factor at redshift z given the present day value of omega_m. Uses the expression
    from Heath1977

    Assumes Flat LCDM cosmology, which is fine given this is also assumed in CambGenerator. Possible improvement
    could be to tabulate this using the CambGenerator so that it would be self consistent for non-LCDM cosmologies.

    :param omega_m: the matter density at the present day
    :param z: the redshift we want the matter density at
    :return: the unnormalised growth factor at redshift z.
    """
    avals = np.logspace(-4.0, np.log10(1.0 / (1.0 + z)), 10000)
    integ = integrate.simps(1.0 / (avals * E_z(omega_m, 1.0 / avals - 1.0)) ** 3, avals, axis=0)
    return 5.0 / 2.0 * omega_m * E_z(omega_m, z) * integ


# TODO: Add options for mnu, h0 default, omega_b, etc
# TODO: Expand to work for smoothing kernels other than Gaussian (perhaps the user can choose Gaussian, Tophat, CIC)
# TODO: Add some basic checks of CAMBGenerator to make sure it is valid
class PTGenerator(object):
    def __init__(self, CAMBGenerator, smooth_type="hinton2017", recon_smoothing_scale=15, mpi_comm=None):
        """ 
        Precomputes certain integrals over the camb power spectrum for efficiency given a list of smoothing scales. Access ks via self.ks, and use get_data for an array
        of all the Perturbation Theory integrals.
        """
        self.logger = logging.getLogger("barry")
        self.smooth_type = smooth_type.lower()
        self.mpi_comm = mpi_comm
        if not validate_smooth_method(smooth_type):
            exit(0)
        self.recon_smoothing_scale = recon_smoothing_scale

        # We should check here that CAMBGenerator points to some pk_lin.
        self.CAMBGenerator = CAMBGenerator

        self.data_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + os.sep + "data/"
        self.filename = self.data_dir + f"PT_{CAMBGenerator.filename_unique}_{self.smooth_type}_{int(self.recon_smoothing_scale * 100)}.pkl"

        self.data = None
        self.logger.info(f"Creating PT data with {self.CAMBGenerator.om_resolution} x {self.CAMBGenerator.h0_resolution}")

    def load_data(self, can_generate=False):
        if not os.path.exists(self.filename):
            if not can_generate:
                msg = "Data does not exist and this isn't the time to generate it!"
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.logger.warning(f"Cannot find PT data at {self.filename}, generating it!")
                self.data = self._generate_data()
        else:
            self.logger.info("Loading existing PT data")
            with open(self.filename, "rb") as f:
                self.data = pickle.load(f)

    @lru_cache(maxsize=512)
    def get_data(self, om=0.31, h0=0.676):
        """ Returns the PT integrals: Sigma, Sigma_dd, Sigma_ss, Sigma_dd,nl, Sigma_sd,nl, Sigma_ss,nl, Sigma_rs,
            R_1, R_2 and the nonlinear parts of the power spectrum"""
        if self.data is None:
            self.load_data()
        omch2 = (om - self.CAMBGenerator.omega_b) * h0 * h0
        data = self._interpolate(omch2, h0)
        return data

    def _generate_points(self, indexes):
        self.logger.info("Generating PT data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Compute the smoothing kernels (assumes a Gaussian smoothing kernel)
        smoothing_kernel = np.exp(-self.CAMBGenerator.ks ** 2 * self.recon_smoothing_scale ** 2 / 2.0)

        # Run CAMBGenerator.get_data once to ensure the data is loaded under CAMBGenerator.data
        _, _, _ = self.CAMBGenerator.get_data()

        # Generate a grid of values for R1, R2, Imn and Jmn
        nx = 200
        xs = np.linspace(-0.999, 0.999, nx)
        r = np.outer(1.0 / self.CAMBGenerator.ks, self.CAMBGenerator.ks)
        R1 = -(1.0 + r ** 2) / (24.0 * r ** 2) * (3.0 - 14.0 * r ** 2 + 3.0 * r ** 4) + (r ** 2 - 1.0) ** 4 / (16.0 * r ** 3) * np.log(
            np.fabs((1.0 + r) / (1.0 - r))
        )
        R2 = (1.0 - r ** 2) / (24.0 * r ** 2) * (3.0 - 2.0 * r ** 2 + 3.0 * r ** 4) + (r ** 2 - 1.0) ** 3 * (1.0 + r ** 2) / (16.0 * r ** 3) * np.log(
            np.fabs((1.0 + r) / (1.0 - r))
        )
        J00 = (
            12.0 / r ** 2
            - 158.0
            + 100.0 * r ** 2
            - 42.0 * r ** 4
            + 3.0 * (r ** 2 - 1.0) ** 3 * (2.0 + 7.0 * r ** 2) / r ** 3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))
        )
        J01 = (
            24.0 / r ** 2
            - 202.0
            + 56.0 * r ** 2
            - 30.0 * r ** 4
            + 3.0 * (r ** 2 - 1.0) ** 3 * (4.0 + 5.0 * r ** 2) / r ** 3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))
        )
        J11 = 12.0 / r ** 2 - 82.0 + 4.0 * r ** 2 - 6.0 * r ** 4 + 3.0 * (r ** 2 - 1.0) ** 3 * (2.0 + r ** 2) / r ** 3 * np.log(np.fabs((1.0 + r) / (1.0 - r)))

        # We get NaNs in R1, R2 etc., when r = 1.0 (diagonals). We manually set these to the correct values.
        # We also get numerical issues for large/small r, so we set these manually to asymptotic limits
        R1[np.diag_indices(len(self.CAMBGenerator.ks))] = 2.0 / 3.0
        R2[np.diag_indices(len(self.CAMBGenerator.ks))] = 0.0
        J00[np.diag_indices(len(self.CAMBGenerator.ks))] = -88.0
        J01[np.diag_indices(len(self.CAMBGenerator.ks))] = -152.0
        J11[np.diag_indices(len(self.CAMBGenerator.ks))] = -72.0
        index = np.where(r < 1.0e-3)
        R1[index] = 16.0 / 15.0 * r[index] ** 2
        R2[index] = 4.0 / 15.0 * r[index] ** 2
        J00[index] = -168.0
        J01[index] = -168.0
        J11[index] = -56.0
        index = np.where(r > 1.0e2)
        R1[index] = 16.0 / 15.0
        R2[index] = 4.0 / 15.0
        J00[index] = -97.6
        J01[index] = -200.0
        J11[index] = -100.8

        rank = 0 if self.mpi_comm is None else mpi_comm.Get_rank()

        data = {}
        for key in ["sigma", "sigma_dd", "sigma_ss", "sigma_nl", "sigma_dd_nl", "sigma_sd_nl", "sigma_ss_nl", "sigma_dd_rs", "sigma_ss_rs"]:
            data[key] = np.zeros((self.CAMBGenerator.om_resolution, self.CAMBGenerator.h0_resolution))
        for key in ["R1", "R2", "Pdd_spt", "Pdt_spt", "Ptt_spt", "Pdd_halofit", "Pdt_halofit", "Ptt_halofit"]:
            data[key] = np.zeros((self.CAMBGenerator.om_resolution, self.CAMBGenerator.h0_resolution, self.CAMBGenerator.k_num))

        for i, j in indexes:
            omch2 = self.CAMBGenerator.omch2s[i]
            h0 = self.CAMBGenerator.h0s[j]

            self.logger.debug("Rank %d Generating %d:%d  %0.3f  %0.3f" % (rank, i, j, omch2, h0))

            # Get the CAMB power spectrum and spline it
            r_drag = self.CAMBGenerator.data[i, j, 0]
            pk_lin = self.CAMBGenerator.data[i, j, 1 : 1 + self.CAMBGenerator.k_num]
            pk_nonlin_0 = self.CAMBGenerator.data[i, j, 1 + self.CAMBGenerator.k_num : 1 + 2 * self.CAMBGenerator.k_num]
            pk_nonlin_z = self.CAMBGenerator.data[i, j, 1 + 2 * self.CAMBGenerator.k_num :]

            # Get the spherical bessel functions
            j0 = special.jn(0, r_drag * self.CAMBGenerator.ks)
            j2 = special.jn(2, r_drag * self.CAMBGenerator.ks)

            # Get the smoothed linear power spectrum which we need to calculate the
            # BAO damping and SPT integrals used in the Noda2017 model
            om = omch2 / (h0 * h0) + self.CAMBGenerator.omega_b
            pk_smooth_lin = smooth(self.CAMBGenerator.ks, pk_lin, method=self.smooth_type, om=om, h0=h0)
            pk_smooth_nonlin_0 = smooth(self.CAMBGenerator.ks, pk_nonlin_0, method=self.smooth_type, om=om, h0=h0)
            pk_smooth_nonlin_z = smooth(self.CAMBGenerator.ks, pk_nonlin_z, method=self.smooth_type, om=om, h0=h0)
            pk_smooth_spline = interpolate.splrep(self.CAMBGenerator.ks, pk_smooth_lin)

            # Sigma^2
            data["sigma"][i, j] = integrate.simps(pk_lin, self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)

            # Sigma^2_dd, Sigma^2_ss   (Seo2016-LPT model)
            data["sigma_dd"][i, j] = integrate.simps(pk_lin * (1.0 - smoothing_kernel) ** 2, self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)
            data["sigma_ss"][i, j] = integrate.simps(pk_lin * smoothing_kernel ** 2, self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)

            # Sigma^2_nl, Sigma^2_dd,nl, Sigma^2_sd,nl Sigma^2_ss,nl (Ding2018-EFT model)
            data["sigma_nl"][i, j] = integrate.simps(pk_lin * (1.0 - j0), self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)
            data["sigma_dd_nl"][i, j] = integrate.simps(pk_lin * (1.0 - smoothing_kernel) ** 2 * (1.0 - j0), self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)
            data["sigma_sd_nl"][i, j] = integrate.simps(
                pk_lin * (0.5 * (smoothing_kernel ** 2 + (1.0 - smoothing_kernel) ** 2) - j0 * smoothing_kernel * (1.0 - smoothing_kernel)),
                self.CAMBGenerator.ks,
            ) / (6.0 * np.pi ** 2)
            data["sigma_ss_nl"][i, j] = integrate.simps(pk_lin * smoothing_kernel ** 2 * (1.0 - j0), self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)

            # Sigma^2_dd,rs, Sigma^2_ss,rs (Noda2019 model)
            data["sigma_dd_rs"][i, j] = integrate.simps(pk_smooth_lin * (1.0 - j0 + 2.0 * j2), self.CAMBGenerator.ks) / (6.0 * np.pi ** 2)
            data["sigma_ss_rs"][i, j] = integrate.simps(pk_smooth_lin * j2, self.CAMBGenerator.ks) / (2.0 * np.pi ** 2)

            # R_1/P_lin, R_2/P_lin
            data["R1"][i, j, :] = self.CAMBGenerator.ks ** 2 * integrate.simps(pk_lin * R1, self.CAMBGenerator.ks, axis=1) / (4.0 * np.pi ** 2)
            data["R2"][i, j, :] = self.CAMBGenerator.ks ** 2 * integrate.simps(pk_lin * R2, self.CAMBGenerator.ks, axis=1) / (4.0 * np.pi ** 2)

            # I_00/P_sm,lin, I_01/P_sm,lin, I_02/P_sm,lin
            for k, kval in enumerate(self.CAMBGenerator.ks):
                rvals = r[k, 0:]
                rx = np.outer(rvals, xs)
                y = kval * np.sqrt(-2.0 * rx.T + 1.0 + rvals ** 2)
                pk_smooth_interp = interpolate.splev(y, pk_smooth_spline)
                index = np.where(np.logical_and(y < self.CAMBGenerator.k_min, y > self.CAMBGenerator.k_max))
                pk_smooth_interp[index] = 0.0
                IP0 = kval ** 2 * ((-10.0 * rx * xs + 7.0 * xs).T + 3.0 * rvals) / (y ** 2)
                IP1 = kval ** 2 * ((-6.0 * rx * xs + 7.0 * xs).T - rvals) / (y ** 2)
                data["Pdd_spt"][i, j, k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP0 * IP0, xs, axis=0), rvals)
                data["Pdt_spt"][i, j, k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP0 * IP1, xs, axis=0), rvals)
                data["Ptt_spt"][i, j, k] = integrate.simps(pk_smooth_lin * integrate.simps(pk_smooth_interp * IP1 * IP1, xs, axis=0), rvals)
            data["Pdd_spt"][i, j, :] *= self.CAMBGenerator.ks ** 3 / (392.0 * np.pi ** 2) / pk_smooth_lin
            data["Pdt_spt"][i, j, :] *= self.CAMBGenerator.ks ** 3 / (392.0 * np.pi ** 2) / pk_smooth_lin
            data["Ptt_spt"][i, j, :] *= self.CAMBGenerator.ks ** 3 / (392.0 * np.pi ** 2) / pk_smooth_lin

            # Add on k^2[J_00, J_01, J_11] to obtain P_sm,spt/P_sm,L - 1
            data["Pdd_spt"][i, j, :] += self.CAMBGenerator.ks ** 2 * integrate.simps(pk_smooth_lin * J00, self.CAMBGenerator.ks, axis=1) / (1008.0 * np.pi ** 2)
            data["Pdt_spt"][i, j, :] += self.CAMBGenerator.ks ** 2 * integrate.simps(pk_smooth_lin * J01, self.CAMBGenerator.ks, axis=1) / (1008.0 * np.pi ** 2)
            data["Ptt_spt"][i, j, :] += self.CAMBGenerator.ks ** 2 * integrate.simps(pk_smooth_lin * J11, self.CAMBGenerator.ks, axis=1) / (336.0 * np.pi ** 2)

            # Compute the non linear correction to the power spectra using the fitting formulae from Jennings2012
            growth_0, growth_z = Growth_factor_Linder(om, 1.0e-4), Growth_factor_Linder(om, self.CAMBGenerator.redshift)
            cfactor = (growth_z + growth_z ** 2 + growth_z ** 3) / (growth_0 + growth_0 ** 2 + growth_0 ** 3)
            Pdt_0 = (-12483.8 * np.sqrt(pk_smooth_nonlin_0) + 2.554 * pk_smooth_nonlin_0 ** 2) / (1381.29 + 2.540 * pk_smooth_nonlin_0)
            Ptt_0 = (-12480.5 * np.sqrt(pk_smooth_nonlin_0) + 1.824 * pk_smooth_nonlin_0 ** 2) / (2165.87 + 1.796 * pk_smooth_nonlin_0)
            Pdt_z = cfactor ** 2 * (Pdt_0 - pk_smooth_nonlin_0) + pk_smooth_nonlin_z
            Ptt_z = cfactor ** 2 * (Ptt_0 - pk_smooth_nonlin_0) + pk_smooth_nonlin_z
            data["Pdd_halofit"][i, j, :] = pk_smooth_nonlin_z / pk_smooth_lin - 1.0
            data["Pdt_halofit"][i, j, :] = Pdt_z / pk_smooth_lin - 1.0
            data["Ptt_halofit"][i, j, :] = Ptt_z / pk_smooth_lin - 1.0

        return data

    def _generate_data(self):
        omch2s = self.CAMBGenerator.omch2s
        h0s = self.CAMBGenerator.h0s
        all_indexes = [(i, j) for i in range(len(omch2s)) for j in range(len(h0s))]

        if self.mpi_comm is None:
            self.logger.info("Running generation locally")
            delegations = [all_indexes]
            all_results = [self._generate_points(all_indexes)]
            rank = 0
        else:
            self.logger.info("Running generation via MPI")
            size = self.mpi_comm.Get_size()
            delegations = [all_indexes[i::size] for i in range(size)]
            run_indexes = self.mpi_comm.scatter(delegations, root=0)
            results = self._generate_points(run_indexes)
            all_results = self.mpi_comm.gather(results, root=0)
            rank = self.mpi_comm.Get_rank()

        if rank == 0:
            data = all_results[0]
            if len(all_results) > 1:
                for inds, new_data in zip(delegations[1:], all_results[1:]):
                    for key in data.keys():
                        dims = len(data[key].shape)
                        for i, j in inds:
                            if dims == 2:
                                data[key][i, j] = new_data[key][i, j]
                            else:
                                data[key][i, j, :] = new_data[key][i, j, :]

            self.logger.info(f"Saving to {self.filename}")
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].astype(np.float32)

            with open(self.filename, "wb") as f:
                pickle.dump(data, f)
            return data
        else:
            return None

    def _interpolate(self, omch2, h0):
        """ Performs bilinear interpolation on the entire pk array """
        omch2_index = (
            1.0
            * (self.CAMBGenerator.om_resolution - 1)
            * (omch2 - self.CAMBGenerator.omch2s[0])
            / (self.CAMBGenerator.omch2s[-1] - self.CAMBGenerator.omch2s[0])
        )

        if self.CAMBGenerator.h0_resolution == 1:
            h0_index = 0
        else:
            h0_index = (
                1.0 * (self.CAMBGenerator.h0_resolution - 1) * (h0 - self.CAMBGenerator.h0s[0]) / (self.CAMBGenerator.h0s[-1] - self.CAMBGenerator.h0s[0])
            )

        x = omch2_index - np.floor(omch2_index)
        y = h0_index - np.floor(h0_index)

        data = self.data
        result = {}
        for key in data.keys():

            v1 = data[key][int(np.floor(omch2_index)), int(np.floor(h0_index))]  # 00
            v2 = data[key][int(np.ceil(omch2_index)), int(np.floor(h0_index))]  # 01

            if self.CAMBGenerator.h0_resolution == 1:
                result[key] = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y)
            else:
                v3 = data[key][int(np.floor(omch2_index)), int(np.ceil(h0_index))]  # 10
                v4 = data[key][int(np.ceil(omch2_index)), int(np.ceil(h0_index))]  # 11
                result[key] = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y) + v3 * y * (1 - x) + v4 * x * y
        return result


def test_rand_h0const():
    g = CambGenerator()
    PT_g = PTGenerator(g)
    PT_g.load_data()

    def fn():
        g.get_data(np.random.uniform(0.1, 0.2))

    return fn


def test_rand():
    g = CambGenerator()
    g.load_data()

    def fn():
        g.get_data(np.random.uniform(0.1, 0.2), h0=np.random.uniform(60, 80))

    return fn


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.cosmology.camb_generator import CambGenerator, getCambGenerator

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not is_local():
        import argparse

        # Set up command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--om_resolution", type=int, default=101, help="Num of Omch2s to generate")
        parser.add_argument("--h0_resolution", type=int, default=1, help="Num of h0s to generate")
        parser.add_argument("--reconsmoothscale", type=float, default=21.21)
        parser.add_argument("--redshift", type=float, default=21.21)
        parser.add_argument("--om", type=float, default=0.31)
        parser.add_argument("--h0", type=float, default=0.676)
        parser.add_argument("--ob", type=float, default=0.04814)
        parser.add_argument("--ns", type=float, default=0.97)
        args = parser.parse_args()

        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        generator = CambGenerator(
            redshift=args.redshift, om_resolution=args.om_resolution, h0_resolution=args.h0_resolution, h0=args.h0, ob=args.ob, ns=args.ns
        )
        rank = mpi_comm.Get_rank()
        if rank == 0:
            generator.load_data(can_generate=True)
        mpi_comm.Barrier()
        pt_generator = PTGenerator(generator, recon_smoothing_scale=args.reconsmoothscale, mpi_comm=mpi_comm)
        pt_generator.load_data(can_generate=True)

    else:
        import timeit
        import matplotlib.pyplot as plt

        c = {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "reconscale": 15}

        generator = CambGenerator(om_resolution=101, h0_resolution=1, h0=c["h0"], ob=c["ob"], ns=c["ns"], redshift=c["z"])
        generator.load_data(can_generate=True)
        pt_generator = PTGenerator(generator, recon_smoothing_scale=15)
        pt_generator.load_data(can_generate=True)
        pt_generator.get_data()

        n = 1000
        print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))

        pk_lin = generator.get_data(0.3)[1]
        pk_smooth_lin = smooth(generator.ks, pk_lin, method=pt_generator.smooth_type)

        plt.plot(generator.ks, pt_generator.get_data(0.2)["Pdd_spt"], color="b", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Pdd_spt"], color="r", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.3$")
        plt.plot(generator.ks, pt_generator.get_data(0.2)["Pdd_halofit"], color="b", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Pdd_halofit"], color="r", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3$")
        plt.ylabel(r"$P_{\delta \delta}/P_{L} - 1$")
        plt.xlim(0.0, 0.3)
        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.show()

        plt.plot(generator.ks, pt_generator.get_data(0.2)["Pdt_spt"], color="b", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Pdt_spt"], color="r", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.3$")
        plt.plot(generator.ks, pt_generator.get_data(0.2)["Pdt_halofit"], color="b", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Pdt_halofit"], color="r", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3$")
        plt.ylabel(r"$P_{\delta \theta}/P_{L} - 1$")
        plt.xlim(0.0, 0.3)
        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.show()

        plt.plot(generator.ks, pt_generator.get_data(0.2)["Ptt_spt"], color="b", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Ptt_spt"], color="r", linestyle="-", label=r"$\mathrm{SPT}\,\Omega_{m}=0.3$")
        plt.plot(generator.ks, pt_generator.get_data(0.2)["Ptt_halofit"], color="b", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.2$")
        plt.plot(generator.ks, pt_generator.get_data(0.3)["Ptt_halofit"], color="r", linestyle="--", label=r"$\mathrm{Halofit}\,\Omega_{m}=0.3$")
        plt.ylabel(r"$P_{\theta \theta}/P_{L} - 1$")
        plt.xlim(0.0, 0.3)
        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.show()
