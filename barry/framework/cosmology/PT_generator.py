import numpy as np
import inspect
import os
import logging

import sys
sys.path.append("../../../")

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.cosmology.power_spectrum_smoothing import smooth, validate_smooth_method

# TODO: Add options for mnu, h0 default, omega_b, etc
# TODO: Expand to work for smoothing kernels other than Gaussian (perhaps the user can choose Gaussian, Tophat, CIC)
# TODO: Add some basic checks of CAMBGenerator to make sure it is valid
class PTGenerator(object):
    def __init__(self, CAMBGenerator, smooth_type="hinton2017", recon_smoothing_scale=10.0):
        """ 
        Precomputes certain integrals over the camb power spectrum for efficiency given a list of smoothing scales. Access ks via self.ks, and use get_data for an array
        of all the Perturbation Theory integrals.
        """
        self.logger = logging.getLogger("barry")
        self.smooth_type = smooth_type.lower()
        if not validate_smooth_method(smooth_type):
            exit(0)
        self.recon_smoothing_scale = recon_smoothing_scale

        # We should check here that CAMBGenerator points to some pk_lin.
        self.CAMBGenerator = CAMBGenerator

        self.data_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + os.sep + "data/"
        self.filename = self.data_dir + f"PT_{int(self.CAMBGenerator.redshift * 100)}_{self.CAMBGenerator.om_resolution}_{self.CAMBGenerator.h0_resolution}_{self.smooth_type}_{int(self.recon_smoothing_scale * 100)}.npy"

        self.data = None
        self.logger.info(f"Creating PT data with {self.CAMBGenerator.om_resolution} x {self.CAMBGenerator.h0_resolution}")

    def load_data(self):
        if not os.path.exists(self.filename):
            self.data = self._generate_data()
        else:
            self.logger.info("Loading existing PT data")
            self.data = np.load(self.filename)

    def get_data(self, om=0.3121, h0=0.6751):
        """ Returns the PT integrals: Sigma, Sigma_dd, Sigma_ss, Sigma_dd,nl, Sigma_sd,nl, Sigma_ss,nl, Sigma_rs,
            R_1, R_2 and the SPT integrals"""
        if self.data is None:
            self.load_data()
        omch2 = (om - self.CAMBGenerator.omega_b) * h0 * h0
        data = self._interpolate(omch2, h0)
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9:9+self.CAMBGenerator.k_num], data[9+self.CAMBGenerator.k_num:9+2*self.CAMBGenerator.k_num]

    def _generate_data(self):
        self.logger.info("Generating PT data")
        os.makedirs(self.data_dir, exist_ok=True)
        from scipy import integrate, special

        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        smoothing_kernel = np.exp(-self.CAMBGenerator.ks**2*self.recon_smoothing_scale**2/4.0)

        # Run CAMBGenerator.get_data once to ensure the data is loaded under CAMBGenerator.data
        _, _ = self.CAMBGenerator.get_data()

        # Generate a grid of values for R1, R2, Imn and Jmn
        r = np.outer(self.CAMBGenerator.ks,1.0/self.CAMBGenerator.ks)
        R1 = -(1.0 + r**2)/(24.0*r**2)*(3.0 - 14.0*r**2 + 3.0*r**4) + (r**2-1.0)**4/(16.0*r**3)*np.log(np.fabs((1.0+r)/(1.0-r)))
        R2 =  (1.0 - r**2)/(24.0*r**2)*(3.0 -  2.0*r**2 + 3.0*r**4) + (r**2-1.0)**3*(1.0+r**2)/(16.0*r**3)*np.log(np.fabs((1.0+r)/(1.0-r)))

        # We get NaNs in R1/R2 when r = 1.0 (diagonals). We manually set these to the correct values. 
        # We also get numerical issues for large/small r, so we set these manually to asymptotic limits
        R1[np.diag_indices(len(self.CAMBGenerator.ks))] = 2.0/3.0
        R2[np.diag_indices(len(self.CAMBGenerator.ks))] = 0.0
        index = np.where(r < 1.0e-3)
        R1[index] = 16.0/15.0*r[index]**2
        R2[index] = 4.0/15.0*r[index]**2
        index = np.where(r > 1.0e3)
        R1[index] = 16.0/15.0
        R2[index] = 4.0/15.0

        data = np.zeros((self.CAMBGenerator.om_resolution, self.CAMBGenerator.h0_resolution, 9 + 2*self.CAMBGenerator.k_num))
        for i, omch2 in enumerate(self.CAMBGenerator.omch2s):
            for j, h0 in enumerate(self.CAMBGenerator.h0s):
                self.logger.debug("Generating %d:%d  %0.3f  %0.3f" % (i, j, omch2, h0))

                # Get the CAMB power spectrum and spline it
                r_drag, pk_lin = self.CAMBGenerator.data[i, j, 0], self.CAMBGenerator.data[i, j, 1:]

                # Get the spherical bessel functions
                j0 = special.jn(0, r_drag*self.CAMBGenerator.ks)
                j2 = special.jn(2, r_drag*self.CAMBGenerator.ks)

                # Get the smoothed linear power spectrum which we need to calculate the
                # BAO damping and SPT integrals used in the Noda2017 model
                om = omch2/(h0 * h0) + self.CAMBGenerator.omega_b
                pk_smooth_lin = smooth(self.CAMBGenerator.ks, pk_lin, method=self.smooth_type, om=om, h0=h0)

                # Sigma^2
                data[i, j, 0] = integrate.simps(pk_lin,self.CAMBGenerator.ks)/(6.0*np.pi**2)

                # Sigma^2_dd, Sigma^2_ss   (Seo2016-LPT model)
                data[i, j, 1] = integrate.simps(pk_lin*(1.0 - smoothing_kernel)**2,self.CAMBGenerator.ks)/(6.0*np.pi**2)
                data[i, j, 2] = integrate.simps(pk_lin*smoothing_kernel**2,self.CAMBGenerator.ks)/(6.0*np.pi**2)

                # Sigma^2_nl, Sigma^2_dd,nl, Sigma^2_sd,nl Sigma^2_ss,nl (Ding2018-EFT model)
                data[i, j, 3] = integrate.simps(pk_lin*(1.0 - j0),self.CAMBGenerator.ks)/(6.0*np.pi**2)
                data[i, j, 4] = integrate.simps(pk_lin*(1.0 - smoothing_kernel)**2*(1.0 - j0),self.CAMBGenerator.ks)/(6.0*np.pi**2)
                data[i, j, 5] = integrate.simps(pk_lin*(0.5*(smoothing_kernel**2-(1.0 - smoothing_kernel)**2) - j0*smoothing_kernel*(1.0 - smoothing_kernel)),self.CAMBGenerator.ks)/(6.0*np.pi**2)
                data[i, j, 6] = integrate.simps(pk_lin*smoothing_kernel**2*(1.0 - j0),self.CAMBGenerator.ks)/(6.0*np.pi**2)

                # Sigma^2_dd,rs, Sigma^2_ss,rs (Noda2017 model)
                data[i, j, 7] = integrate.simps(pk_smooth_lin*(1.0 - j0 + 2.0*j2),self.CAMBGenerator.ks)/(6.0*np.pi**2)
                data[i, j, 8] = integrate.simps(pk_smooth_lin*j2,self.CAMBGenerator.ks)/(2.0*np.pi**2)

                # R_1/P_lin, R_2/P_lin
                data[i, j, 9:9+self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**3*integrate.simps(pk_lin*R1, self.CAMBGenerator.ks, axis=0)/(4.0*np.pi**2)
                data[i, j, 9+self.CAMBGenerator.k_num:9+2*self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**3*integrate.simps(pk_lin*R2, self.CAMBGenerator.ks, axis=0)/(4.0*np.pi**2)

        self.logger.info(f"Saving to {self.filename}")
        np.save(self.filename, data)
        return data

    def _interpolate(self, omch2, h0):
        """ Performs bilinear interpolation on the entire pk array """
        omch2_index = 1.0 * (self.CAMBGenerator.om_resolution - 1) * (omch2 - self.CAMBGenerator.omch2s[0]) / (self.CAMBGenerator.omch2s[-1] - self.CAMBGenerator.omch2s[0])

        if self.CAMBGenerator.h0_resolution == 1:
            h0_index = 0
        else:
            h0_index = 1.0 * (self.CAMBGenerator.h0_resolution - 1) * (h0 - self.CAMBGenerator.h0s[0]) / (self.CAMBGenerator.h0s[-1] - self.CAMBGenerator.h0s[0])

        x = omch2_index - np.floor(omch2_index)
        y = h0_index - np.floor(h0_index)

        data = self.data
        v1 = data[int(np.floor(omch2_index)), int(np.floor(h0_index))]  # 00
        v2 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index))]   # 01

        if self.CAMBGenerator.h0_resolution == 1:
            final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y)
        else:
            v3 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index))]  # 10
            v4 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index))]  # 11
            final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y) + v3 * y * (1 - x) + v4 * x * y
        return final


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
    sys.path.append("../../..")
    from barry.framework.cosmology.camb_generator import CambGenerator

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    generator = CambGenerator()
    PT_generator = PTGenerator(generator)

    import timeit
    n = 10000
    print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))
    import matplotlib.pyplot as plt

    nvals = 100
    sigmas = np.empty((nvals, 9))
    oms = np.linspace(0.2,0.4,nvals)
    for i, om in enumerate(oms):
        sigmas[i] = PT_generator.get_data(om)[0:9]

    plt.plot(oms, sigmas[0:, 0], label=r"$\Sigma^{2}$")
    plt.plot(oms, sigmas[0:, 1], label=r"$\Sigma^{2}_{dd}$")
    plt.plot(oms, sigmas[0:, 2], label=r"$\Sigma^{2}_{ss}$")
    plt.plot(oms, sigmas[0:, 3], label=r"$\Sigma^{2}_{nl}$")
    plt.plot(oms, sigmas[0:, 4], label=r"$\Sigma^{2}_{dd,nl}$")
    plt.plot(oms, sigmas[0:, 5], label=r"$\Sigma^{2}_{sd,nl}$")
    plt.plot(oms, sigmas[0:, 6], label=r"$\Sigma^{2}_{ss,nl}$")
    plt.plot(oms, sigmas[0:, 7], label=r"$\Sigma^{2}_{dd,rs}$")
    plt.plot(oms, sigmas[0:, 8], label=r"$\Sigma^{2}_{ss,rs}$")
    plt.ylim(0.0, 50.0)
    plt.legend()
    plt.show()

    plt.plot(PT_generator.CAMBGenerator.ks, PT_generator.get_data(0.2)[9], label=r"$R_{1}(\Omega_{m}=0.2)$")
    plt.plot(PT_generator.CAMBGenerator.ks, PT_generator.get_data(0.3)[9], label=r"$R_{1}(\Omega_{m}=0.3)$")
    plt.plot(PT_generator.CAMBGenerator.ks, -PT_generator.get_data(0.2)[10], label=r"$-R_{2}(\Omega_{m}=0.2)$")
    plt.plot(PT_generator.CAMBGenerator.ks, -PT_generator.get_data(0.3)[10], label=r"$-R_{2}(\Omega_{m}=0.3)$")
    plt.ylim(1.0e-8, 1.0e5)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

    growth = 0.3**0.55
    part1 = 1.0 + 3.0 / 7.0 * ((PT_generator.get_data(0.3)[9] * (1.0 - 4.0 / 9.0)) + PT_generator.get_data(0.3)[10])
    part2 = growth + 3.0 / 7.0 * growth * PT_generator.get_data(0.3)[9] * (2.0 - 1.0 / (3.0)) + 6.0 / 7.0 * growth * PT_generator.get_data(0.3)[10]
    damping = np.exp(-PT_generator.CAMBGenerator.ks**2*(1.0 + (2.0 + growth)*growth)*PT_generator.get_data(0.3)[0]/2.0)

    #plt.plot(PT_generator.CAMBGenerator.ks, generator.get_data(0.3)[1])
    plt.plot(PT_generator.CAMBGenerator.ks, (part1+part2)**2*damping**2)
    plt.plot(PT_generator.CAMBGenerator.ks, (1.0+growth)**2*damping**2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1.0e-2, 0.4)
    plt.ylim(1.0e-3, 15.0)
    plt.legend()
    plt.show()
