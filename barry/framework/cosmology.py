import numpy as np
import camb
import inspect
import os


class CambGenerator(object):
    def __init__(self):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + os.sep + "data/"
        self.filename = self.data_dir + "cosmo.npy"

        self.redshift = 0.1

        self.k_min = 1e-4
        self.k_max = 2
        self.k_num = 200
        self.ks = np.logspace(np.log(self.k_min), np.log(self.k_max), self.k_num, base=np.e)

        self.resolution = 10
        self.omch2s = np.linspace(0.1, 0.2, self.resolution)
        self.h0s = np.linspace(60, 80, self.resolution)

        self.data = None

    def load_data(self):
        if not os.path.exists(self.filename):
            self.data = self._generate_data()
        else:
            self.data = np.load(self.filename)

    def get_data(self, omch2, h0):
        if self.data is None:
            self.load_data()
        return self._interpolate(omch2, h0)

    def _generate_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        pars = camb.CAMBparams()
        pars.set_dark_energy()
        pars.InitPower.set_params(ns=0.965)
        redshifts = [self.redshift]
        pars.set_matter_power(redshifts=redshifts, kmax=self.k_max)
        pars.NonLinear = camb.model.NonLinear_both

        data = np.zeros((self.resolution, self.resolution, self.k_num))
        for i, omch2 in enumerate(self.omch2s):
            for j, h0 in enumerate(self.h0s):
                print("Generating %d:%d" % (i, j))
                pars.set_cosmology(H0=h0, omch2=omch2)
                results = camb.get_results(pars)
                results.calc_power_spectra(pars)
                kh, z, pk = results.get_matter_power_spectrum(minkh=self.k_min, maxkh=self.k_max, npoints=self.k_num)
                data[i, j, :] = pk

        data = data.astype(np.float32)
        np.save(self.filename, data)
        return data

    def _interpolate(self, omch2, h0):
        """ Performs bilinear interpolation on the entire pk array """
        omch2_index = 1.0 * (self.resolution - 1) * (omch2 - self.omch2s[0]) / (self.omch2s[-1] - self.omch2s[0])
        h0_index = 1.0 * (self.resolution - 1) * (h0 - self.h0s[0]) / (self.h0s[-1] - self.h0s[0])

        x = omch2_index - np.floor(omch2_index)
        y = h0_index - np.floor(h0_index)

        data = self.data

        v1 = data[int(np.floor(omch2_index)), int(np.floor(h0_index))]  # 00
        v2 = data[int(np.ceil(omch2_index)), int(np.floor(h0_index))]   # 01
        v3 = data[int(np.floor(omch2_index)), int(np.ceil(h0_index))]   # 10
        v4 = data[int(np.ceil(omch2_index)), int(np.ceil(h0_index))]    # 11

        final = v1 * (1 - x) * (1 - y) + v2 * x * (1 - y) + v3 * y * (1 - x) + v4 * x * y
        return final


def test_rand():
    g = CambGenerator()
    g.load_data()

    def fn():
        g.get_data(np.random.uniform(0.1, 0.2), np.random.uniform(60, 80))

    return fn

if __name__ == "__main__":

    generator = CambGenerator()

    import timeit
    n = 10000
    print("Takes on average, %.1f microseconds" % (timeit.timeit(test_rand(), number=n) * 1e6 / n))

    # import matplotlib.pyplot as plt
    # plt.plot(generator.ks, generator.get_data(0.15, 70))
    # plt.plot(generator.ks, generator.get_data(0.16, 70))
    # plt.plot(generator.ks, generator.get_data(0.15, 71))
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
