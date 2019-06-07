import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.postprocessing import PureBAOExtractor


def calc_cov_noda(pk_cov, denoms, pks):
    # Implementing equation 23 of arXiv:1901.06854v1
    # Yes, super slow non-vectorised to make sure its exactly as described
    num = pk_cov.shape[0]
    cov = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            prefactor = 1 / (denoms[i] * denoms[j] * (pks[i] * pks[j])**2)

            # Here is our first issue. The paper does not define over what ranges m and n are summed
            # so I will assume all valid values
            sum = 0
            for m in range(num):
                for n in range(num):
                    sum += pks[m] * pks[n] * pk_cov[i, j] \
                           - pks[m] * pks[j] * pk_cov[i, n] \
                           - pks[i] * pks[n] * pk_cov[m, j] \
                           + pks[i] * pks[j] * pk_cov[m, n]
            cov[i, j] = prefactor * sum
        print(i)
    return np.array(cov)

if __name__ == "__main__":

    camb = CambGenerator()
    r_s, _ = camb.get_data()
    extractor = PureBAOExtractor(r_s)

    data = MockPowerSpectrum()
    data2 = MockPowerSpectrum(postprocess=extractor)

    ks = data.ks
    pk = data.data
    pk_cov = data.cov
    denoms = data2.postprocess.postprocess(ks, pk, return_denominator=True)

    cov_brute = data2.cov

    cov_noda = calc_cov_noda(pk_cov, denoms, pk)

    fig, axes = plt.subplots(nrows=3, figsize=(4, 9))
    sb.heatmap(cov_brute, ax=axes[0])
    sb.heatmap(cov_noda, ax=axes[1])
    sb.heatmap(cov_brute - cov_noda, ax=axes[2])
    plt.show()
