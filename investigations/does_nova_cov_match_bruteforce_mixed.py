import numpy as np
from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.datasets import MockPowerSpectrum
from barry.framework.postprocessing import BAOExtractor, PureBAOExtractor


def get_rs_rs_index(pk_cov, denoms, ks, pks, delta_k, i, j):
    prefactor = 1 / (denoms[i] * denoms[j] * (pks[i] * pks[j]) ** 2)

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
    return sum * prefactor


def get_rs_ps_index(pk_cov, denoms, ks, pks, delta_k, i, j):
    prefactor = 1 / (denoms[i] * (pks[i] ** 2))
    valid_l = np.where(np.abs(ks - ks[i]) < delta_k)[0]
    sum = 0
    for l in valid_l:
        sum += pks[l] * pk_cov[i, j] - pks[i] * pk_cov[l, j]
    return sum * prefactor


def calc_cov_noda(pk_cov, denoms, ks, pks, delta_k, is_extracted):
    # Implementing equation 23 of arXiv:1901.06854v1
    # Yes, super slow non-vectorised to make sure its exactly as described
    num = pk_cov.shape[0]
    cov = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if is_extracted[i] and is_extracted[j]:
                cov[i, j] = get_rs_rs_index(pk_cov, denoms, ks, pks, delta_k, i, j)
            elif is_extracted[i] and not is_extracted[j]:
                cov[i, j] = get_rs_ps_index(pk_cov, denoms, ks, pks, delta_k, i, j)
            elif not is_extracted[i] and is_extracted[j]:
                cov[i, j] = get_rs_ps_index(pk_cov, denoms, ks, pks, delta_k, j, i)
            else:
                cov[i, j] = pk_cov[i, j]
    return cov


if __name__ == "__main__":
    import seaborn as sb
    import matplotlib.pyplot as plt

    camb = CambGenerator()
    r_s, _ = camb.get_data()
    extractor = BAOExtractor(r_s, reorder=False)
    extractor2 = PureBAOExtractor(r_s)

    step_size = 3
    data = MockPowerSpectrum(step_size=step_size, fake_diag=False, apply_hartlap_correction=False)
    data3 = MockPowerSpectrum(step_size=step_size, fake_diag=True, apply_hartlap_correction=False)
    data2 = MockPowerSpectrum(postprocess=extractor, step_size=step_size, apply_hartlap_correction=False)

    ks = data.ks
    pk = data.data
    pk_cov = data.cov
    pk_cov_diag = data3.cov
    denoms = extractor2.postprocess(ks, pk, return_denominator=True)
    is_extracted = extractor.get_is_extracted(ks)
    cov_brute = data2.cov

    k_range = extractor.get_krange()
    cov_noda = calc_cov_noda(pk_cov, denoms, ks, pk, k_range, is_extracted)
    cov_noda_diag = calc_cov_noda(pk_cov_diag, denoms, ks, pk, k_range, is_extracted)

    la_cov_brute = np.log(np.abs(cov_brute))
    la_cov_noda = np.log(np.abs(cov_noda) + np.abs(cov_brute).min())
    la_cov_noda_diag = np.log(np.abs(cov_noda_diag) + np.abs(cov_brute).min())

    fig, axes = plt.subplots(ncols=4, figsize=(18, 4))
    axes[0].set_title("cov from Mocks")
    axes[1].set_title("cov from Nishimichi 2018, eq 7")
    axes[2].set_title("cov from Nishimichi 2018, eq 7 and diag")
    axes[3].set_title("norm diff first two, capped at unity")
    vmin = min(la_cov_brute.min(), la_cov_noda.min(), la_cov_noda_diag.min())
    vmax = max(la_cov_brute.max(), la_cov_noda.max(), la_cov_noda_diag.max())
    sb.heatmap(la_cov_brute, ax=axes[0], vmin=vmin, vmax=vmax)
    sb.heatmap(la_cov_noda, ax=axes[1], vmin=vmin, vmax=vmax)
    sb.heatmap(la_cov_noda_diag, ax=axes[2], vmin=vmin, vmax=vmax)
    sb.heatmap((cov_brute - cov_noda) / cov_brute, ax=axes[3], vmin=-1, vmax=1)
    fig.subplots_adjust(hspace=0.0)
    plt.show()

    # The interesting finding here is that the covariance in P(k) significantly modifies
    # even the diagonal covariance in the BAO extractor method. Because the extractor utilises
    # a filter of pk values, correlation of pk enters directly into the diagonal error.
    # The paper versions of Noda report better constraints on alpha primarily because of this.