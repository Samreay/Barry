import numpy as np
import inspect
import os
import logging


# TODO: Add options for mnu, h0 default, omega_b, etc
# TODO: Calculate/Tabulate r_s alongside power spectra for different omega_m and hubble. We need this for eh98 smoothing of powerspectra
# TODO: Expand to work for smoothing kernels other than Gaussian (perhaps the user can choose Gaussian, Tophat, CIC)
class PTGenerator(object):
    def __init__(self, CAMBGenerator, smoothing_scale=10.0):
        """ 
        Precomputes certain integrals over the camb power spectrum for efficiency given a list of smoothing scales. Access ks via self.ks, and use get_data for an array
        of all the Perturbation Theory integrals.
        """
        self.logger = logging.getLogger("barry")
        self.smoothing_scale = smoothing_scale

        # We should check here that CAMBGenerator points to some pk_lin.
        self.CAMBGenerator = CAMBGenerator

        self.data_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + os.sep + "data/"
        self.filename = self.data_dir + f"PT_{int(self.CAMBGenerator.redshift * 100)}_{self.CAMBGenerator.om_resolution}_{self.CAMBGenerator.h0_resolution}_{int(self.smoothing_scale * 100)}.npy"

        self.data = None
        self.logger.info(f"Creating PT data with {self.CAMBGenerator.om_resolution} x {self.CAMBGenerator.h0_resolution}")

    def load_data(self):
        if not os.path.exists(self.filename):
            self.data = self._generate_data()
        else:
            self.logger.info("Loading existing PT data")
            self.data = np.load(self.filename)

    def get_data(self, om=0.3121, h0=0.6751):
        """ Returns the PT integrals: Sigma, Sigma_dd, Sigma_ss, R_1, R_1^s, R_2, R_2^s, R_2^d """
        if self.data is None:
            self.load_data()
        omch2 = (om - self.CAMBGenerator.omega_b) * h0 * h0
        data = self._interpolate(omch2, h0)
        return data[0], data[1], data[2], data[3:3+self.CAMBGenerator.k_num], data[3+self.CAMBGenerator.k_num:3+2*self.CAMBGenerator.k_num], data[3+2*self.CAMBGenerator.k_num:3+3*self.CAMBGenerator.k_num], data[3+3*self.CAMBGenerator.k_num:3+4*self.CAMBGenerator.k_num], data[3+4*self.CAMBGenerator.k_num:]

    def _generate_data(self):
        self.logger.info("Generating PT data")
        os.makedirs(self.data_dir, exist_ok=True)
        from scipy import integrate, interpolate

        # Compute the smoothing kernel (assumes a Gaussian smoothing kernel)
        smoothing_kernel = np.exp(-self.CAMBGenerator.ks**2*self.smoothing_scale**2/4.0)
        smoothing_kernel_spline = interpolate.splrep(self.CAMBGenerator.ks, smoothing_kernel)

        # Run CAMBGenerator.get_data once to ensure the data is loaded under CAMBGenerator.data
        pk_lin = self.CAMBGenerator.get_data() 

        # Generate a grid of values for R1 and R2
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

        data = np.zeros((self.CAMBGenerator.om_resolution, self.CAMBGenerator.h0_resolution, 3 + 5*self.CAMBGenerator.k_num))
        for i, omch2 in enumerate(self.CAMBGenerator.omch2s):
            for j, h0 in enumerate(self.CAMBGenerator.h0s):
                self.logger.debug("Generating %d:%d  %0.3f  %0.3f" % (i, j, omch2, h0))

                # Get the CAMB power spectrum and spline it
                pk_lin = self.CAMBGenerator.data[i,j,0:]
                pk_lin_spline = interpolate.splrep(self.CAMBGenerator.ks,pk_lin)

                # Sigma^2
                data[i, j, 0] = integrate.simps(pk_lin,self.CAMBGenerator.ks)/(3.0*np.pi**2)

                # Sigma^2_dd, Sigma^2_ss
                data[i, j, 1] = integrate.simps(pk_lin*(1.0 - smoothing_kernel)**2,self.CAMBGenerator.ks)/(3.0*np.pi**2)
                data[i, j, 2] = integrate.simps(pk_lin*smoothing_kernel**2,self.CAMBGenerator.ks)/(3.0*np.pi**2)

                # R_1, R_1^s
                data[i, j, 3:3+self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**2*pk_lin*integrate.simps(pk_lin*R1, axis=0)/(4.0*np.pi**2)
                data[i, j, 3+self.CAMBGenerator.k_num:3+2*self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**2*pk_lin*integrate.simps(pk_lin*R1*smoothing_kernel, axis=0)/(4.0*np.pi**2)

                # R_2, R_2^s, R_2^d
                data[i, j, 3+2*self.CAMBGenerator.k_num:3+3*self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**2*pk_lin*integrate.simps(pk_lin*R2, axis=0)/(4.0*np.pi**2)
                data[i, j, 3+3*self.CAMBGenerator.k_num:3+4*self.CAMBGenerator.k_num] = self.CAMBGenerator.ks**2*pk_lin*integrate.simps(pk_lin*R2*smoothing_kernel, axis=0)/(4.0*np.pi**2)
                data[i, j, 3+4*self.CAMBGenerator.k_num:] = self.CAMBGenerator.ks**2*pk_lin*integrate.simps(pk_lin*R2*(1.0-smoothing_kernel), axis=0)/(4.0*np.pi**2)

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
    PT_generator_10 = PTGenerator(generator)
    PT_generator_15 = PTGenerator(generator, smoothing_scale=15.0)

    import timeit
    n = 10000
    print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand_h0const(), number=n) * 1e6 / n))
    import matplotlib.pyplot as plt

    plt.plot(PT_generator_10.CAMBGenerator.ks, PT_generator_10.get_data(0.3)[3])
    plt.plot(PT_generator_10.CAMBGenerator.ks, PT_generator_10.get_data(0.3)[4])
    plt.plot(PT_generator_15.CAMBGenerator.ks, PT_generator_15.get_data(0.2)[4])
    plt.ylim(1.0e-5, 1.0e8)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    plt.plot(PT_generator_10.CAMBGenerator.ks, np.fabs(PT_generator_10.get_data(0.3)[5]))
    plt.plot(PT_generator_10.CAMBGenerator.ks, np.fabs(PT_generator_10.get_data(0.3)[6]))
    plt.plot(PT_generator_10.CAMBGenerator.ks, np.fabs(PT_generator_10.get_data(0.3)[7]))
    plt.plot(PT_generator_15.CAMBGenerator.ks, np.fabs(PT_generator_15.get_data(0.2)[5]))
    plt.ylim(1.0e-5, 1.0e7)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
