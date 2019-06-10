import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.postprocessing import PureBAOExtractor


def calc_cov_noda(pk_cov, denoms, ks, pks, delta_k):
    # Implementing equation 23 of arXiv:1901.06854v1
    # Yes, super slow non-vectorised to make sure its exactly as described
    num = pk_cov.shape[0]
    cov = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            prefactor = 1 / (denoms[i] * denoms[j] * (pks[i] * pks[j])**2)

            # Here is our first issue. The paper 1901.06854 does not define m or n,
            # however, from 1708.00375v3 we have that it is the values within k_range
            valid_m = np.where(np.abs(ks - ks[i]) < delta_k)[0]
            valid_n = np.where(np.abs(ks - ks[j]) < delta_k)[0]
            sum = 0
            for m in valid_m:
                for n in valid_n:
                    sum += pks[m] * pks[n] * pk_cov[i, j] \
                           - pks[m] * pks[j] * pk_cov[i, n] \
                           - pks[i] * pks[n] * pk_cov[m, j] \
                           + pks[i] * pks[j] * pk_cov[m, n]
            cov[i, j] = prefactor * sum
    return np.array(cov)


if __name__ == "__main__":

    camb = CambGenerator()
    r_s, _ = camb.get_data()
    extractor = PureBAOExtractor(r_s)

    step_size = 3
    data = MockPowerSpectrum(step_size=step_size)
    data2 = MockPowerSpectrum(postprocess=extractor, step_size=step_size)

    ks = data.ks
    pk = data.data
    pk_cov = data.cov
    denoms = data2.postprocess.postprocess(ks, pk, return_denominator=True)

    cov_brute = data2.cov

    k_range = extractor.get_krange()
    cov_noda = calc_cov_noda(pk_cov, denoms, ks, pk, k_range)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    axes[0].set_title("Covariance from Mocks")
    axes[1].set_title("Covariance from Nishimichi 2018, eq 7")
    axes[2].set_title("Normalised difference, capped at unity")
    vmin, vmax = -0.002, 0.009
    sb.heatmap(cov_brute, ax=axes[0])
    sb.heatmap(cov_noda, ax=axes[1])
    sb.heatmap((cov_brute - cov_noda) / cov_brute, ax=axes[2], vmin=-1, vmax=1)
    plt.show()

    # The interesting finding here is that the covariance in P(k) significantly modifies
    # even the diagonal covariance in the BAO extractor method. Because the extractor utilises
    # a filter of pk values, correlation of pk enters directly into the diagonal error.
    # The paper versions of Noda report better constraints on alpha primarily because of this.