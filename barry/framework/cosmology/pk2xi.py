import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d


def pk2xiGauss(ks, pks, ss, interpolateDetail=3, a=0.5):
    ks2 = np.logspace(np.log(np.min(ks)), np.log(np.max(ks)), interpolateDetail * ks.size, base=np.e)
    pks2 = interp1d(ks, pks, kind='linear')(ks2)

    # Set up output array
    xis = np.zeros(ss.size)

    # Precompute k^2 P(k)and gauss (note missing a ks factor below because integrating in log space)
    kkpks = ks2 * pks2 * np.exp(-ks2 * ks2 * a * a)

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        integrand = kkpks * np.sin(ks2 * s) / s
        xis[i] = simps(integrand, ks2)
    xis /= (2 * np.pi * np.pi)
    return xis

if __name__ == "__main__":
    # TODO Test this method
    pass