import numpy as np
import inspect
import os
import logging

import sys
sys.path.append("../../../")

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
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9:9+self.CAMBGenerator.k_num], data[9+self.CAMBGenerator.k_num:9+2*self.CAMBGenerator.k_num], data[9+2*self.CAMBGenerator.k_num:9+3*self.CAMBGenerator.k_num], data[9+3*self.CAMBGenerator.k_num:9+4*self.CAMBGenerator.k_num], data[9+4*self.CAMBGenerator.k_num:9+5*self.CAMBGenerator.k_num], data[9+5*self.CAMBGenerator.k_num:9+6*self.CAMBGenerator.k_num], data[9+6*self.CAMBGenerator.k_num:9+7*self.CAMBGenerator.k_num], data[9+7*self.CAMBGenerator.k_num:9+8*self.CAMBGenerator.k_num]

    def _generate_data(self):
        self.logger.info("Generating PT data")
        os.makedirs(self.data_dir, exist_ok=True)
        from scipy import integrate, special, interpolate

        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        smoothing_kernel = np.exp(-self.CAMBGenerator.ks**2*self.recon_smoothing_scale**2/4.0)

        # Run CAMBGenerator.get_data once to ensure the data is loaded under CAMBGenerator.data
        _, _ = self.CAMBGenerator.get_data()

        # Generate a grid of values for R1, R2, Imn and Jmn
        nx = 200
        xs = np.linspace(-0.999, 0.999, nx)
        r = np.outer(self.CAMBGenerator.ks,1.0/self.CAMBGenerator.ks)
        R1 = -(1.0 + r**2)/(24.0*r**2)*(3.0 - 14.0*r**2 + 3.0*r**4) + (r**2-1.0)**4/(16.0*r**3)*np.log(np.fabs((1.0+r)/(1.0-r)))
        R2 =  (1.0 - r**2)/(24.0*r**2)*(3.0 -  2.0*r**2 + 3.0*r**4) + (r**2-1.0)**3*(1.0+r**2)/(16.0*r**3)*np.log(np.fabs((1.0+r)/(1.0-r)))
        J00 = (12.0/r**2 - 158.0 + 100.0*r**2 - 42.0*r**4 + 3.0*(r**2-1.0)**3*(2.0+7.0*r**2)/r**3*np.log(np.fabs((1.0+r)/(1.0-r))))/3024.0
        J01 = (24.0/r**2 - 202.0 +  56.0*r**2 - 30.0*r**4 + 3.0*(r**2-1.0)**3*(4.0+5.0*r**2)/r**3*np.log(np.fabs((1.0+r)/(1.0-r))))/3024.0
        J11 = (12.0/r**2 -  82.0 +   4.0*r**2 -  6.0*r**4 + 3.0*(r**2-1.0)**3*(2.0+    r**2)/r**3*np.log(np.fabs((1.0+r)/(1.0-r))))/1008.0

        # We get NaNs in R1, R2 etc., when r = 1.0 (diagonals). We manually set these to the correct values.
        # We also get numerical issues for large/small r, so we set these manually to asymptotic limits
        R1[np.diag_indices(len(self.CAMBGenerator.ks))] = 2.0/3.0
        R2[np.diag_indices(len(self.CAMBGenerator.ks))] = 0.0
        J00[np.diag_indices(len(self.CAMBGenerator.ks))] = -11.0/378.0
        J01[np.diag_indices(len(self.CAMBGenerator.ks))] = -19.0/378.0
        J11[np.diag_indices(len(self.CAMBGenerator.ks))] = -27.0/378.0
        index = np.where(r < 1.0e-3)
        R1[index] = 16.0/15.0*r[index]**2
        R2[index] = 4.0/15.0*r[index]**2
        J00[index] = -168.0/3024.0
        J01[index] = -168.0/3024.0
        J11[index] = -168.0/3024.0
        index = np.where(r > 1.0e3)
        R1[index] = 16.0/15.0
        R2[index] = 4.0/15.0
        J00[index] = -97.6/3024.0
        J01[index] = -200.0/3024.0
        J11[index] = -1.0/10.0

        data = np.zeros((self.CAMBGenerator.om_resolution, self.CAMBGenerator.h0_resolution, 9 + 8*self.CAMBGenerator.k_num))
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
                pk_smooth_spline = interpolate.splrep(self.CAMBGenerator.ks, pk_smooth_lin)

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

                # J_00, J_01, J_11
                data[i, j, 9+2*self.CAMBGenerator.k_num:9+3*self.CAMBGenerator.k_num] = 3.0*self.CAMBGenerator.ks**3*integrate.simps(pk_smooth_lin*J00, self.CAMBGenerator.ks, axis=0)/(np.pi**2)
                data[i, j, 9+3*self.CAMBGenerator.k_num:9+4*self.CAMBGenerator.k_num] = 3.0*self.CAMBGenerator.ks**3*integrate.simps(pk_smooth_lin*J01, self.CAMBGenerator.ks, axis=0)/(np.pi**2)
                data[i, j, 9+4*self.CAMBGenerator.k_num:9+5*self.CAMBGenerator.k_num] = 3.0*self.CAMBGenerator.ks**3*integrate.simps(pk_smooth_lin*J11, self.CAMBGenerator.ks, axis=0)/(np.pi**2)

                # I_00/P_sm,lin, I_01/P_sm,lin, I_02/P_sm,lin
                for k, kval in enumerate(self.CAMBGenerator.ks):
                    rvals = r[0:, k]
                    rx = np.outer(rvals, xs)
                    y = kval * np.sqrt(-2.0*rx.T + 1.0 + rvals**2)
                    pk_smooth_interp = interpolate.splev(y, pk_smooth_spline)
                    index = np.where(np.logical_and(y<self.CAMBGenerator.k_min,y>self.CAMBGenerator.k_max))
                    pk_smooth_interp[index] = 0.0
                    r2pk = rvals ** 2 * pk_smooth_lin
                    IP0 = kval**2*((-10.0*rx*xs + 7.0*xs).T + 3.0*rvals) / (14.0 * rvals * y**2)
                    IP1 = kval**2*((- 6.0*rx*xs + 7.0*xs).T - rvals) / (14.0 * rvals * y**2)
                    data[i, j, 9 + 5 * self.CAMBGenerator.k_num + k] = integrate.simps(r2pk * integrate.simps(pk_smooth_interp*IP0*IP0, xs, axis=0), rvals)
                    data[i, j, 9 + 6 * self.CAMBGenerator.k_num + k] = integrate.simps(r2pk * integrate.simps(pk_smooth_interp*IP0*IP1, xs, axis=0), rvals)
                    data[i, j, 9 + 7 * self.CAMBGenerator.k_num + k] = integrate.simps(r2pk * integrate.simps(pk_smooth_interp*IP1*IP1, xs, axis=0), rvals)
                data[i, j, 9 + 5 * self.CAMBGenerator.k_num:9 + 6 * self.CAMBGenerator.k_num] *= self.CAMBGenerator.ks**3/(2.0*np.pi**2)/pk_smooth_lin
                data[i, j, 9 + 6 * self.CAMBGenerator.k_num:9 + 7 * self.CAMBGenerator.k_num] *= self.CAMBGenerator.ks**3/(2.0*np.pi**2)/pk_smooth_lin
                data[i, j, 9 + 7 * self.CAMBGenerator.k_num:9 + 8 * self.CAMBGenerator.k_num] *= self.CAMBGenerator.ks**3/(2.0*np.pi**2)/pk_smooth_lin

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
    import pandas as pd

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)15s]   %(message)s")
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    generator = CambGenerator(om_resolution=20)
    PT_generator = PTGenerator(generator, recon_smoothing_scale=21.21)

    #import timeit
    #n = 1000
    #print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))

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

    PT = np.array(
        pd.read_csv("/Volumes/Work/ICRAR/codes/PT_integrals_waves_linear.dat", delim_whitespace=True, dtype=float,
                    header=None, skiprows=1))
    PT = PT.T
    pk_dd = PT[1,0:] + 2.0*PT[2,0:] + 6.0*PT[0,0:]**2*PT[1,0:]*PT[18,0:]
    pk_dt = PT[1,0:] + 2.0*PT[3,0:] + 6.0*PT[0,0:]**2*PT[1,0:]*PT[19,0:]
    pk_tt = PT[1,0:] + 2.0*PT[7,0:] + 6.0*PT[0,0:]**2*PT[1,0:]*PT[22,0:]


    pk_lin = generator.get_data(0.3121)[1]
    pk_smooth_lin = smooth(generator.ks, pk_lin, method=PT_generator.smooth_type)
    plt.plot(PT_generator.CAMBGenerator.ks, generator.get_data(0.3121)[1], label=r"$P(k)$")
    plt.plot(PT[0,0:], -6.0*PT[0,0:]**2*PT[1,0:]*PT[18,0:], label=r"$P_{PT,\delta \delta}$")
    plt.plot(PT[0,0:], -6.0*PT[0,0:]**2*PT[1,0:]*PT[19,0:], label=r"$P_{PT,\delta \theta}$")
    plt.plot(PT[0,0:], -6.0*PT[0,0:]**2*PT[1,0:]*PT[22,0:], label=r"$P_{PT,\theta \theta}$")
    plt.plot(PT_generator.CAMBGenerator.ks, -pk_smooth_lin*(PT_generator.get_data(0.3121)[11]), label=r"$P_{sm,\delta \delta}(\Omega_{m}=0.3)$")
    plt.plot(PT_generator.CAMBGenerator.ks, -pk_smooth_lin*(PT_generator.get_data(0.3121)[12]), label=r"$P_{sm,\delta \theta}(\Omega_{m}=0.3)$")
    plt.plot(PT_generator.CAMBGenerator.ks, -pk_smooth_lin*(PT_generator.get_data(0.3121)[13]), label=r"$P_{sm,\theta \theta}(\Omega_{m}=0.3)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()