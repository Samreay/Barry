import numpy as np
import logging
from barry.cosmology.camb_generator import getCambGenerator
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import PureBAOExtractor


def calc_cov_noda(pk_cov, denoms, ks, pks, delta_k, assume_diag=False):
    if assume_diag:
        pk_cov = np.diag(np.diag(pk_cov))
    # Implementing equation 23 of arXiv:1901.06854v1
    # Yes, super slow non-vectorised to make sure its exactly as described
    num = pk_cov.shape[0]
    cov = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            prefactor = 1.0 / (denoms[i] * denoms[j] * (pks[i] * pks[j]) ** 2)

            # Here is our first issue. The paper 1901.06854 does not define m or n,
            # however, from 1708.00375v3 we have that it is the values within k_range
            valid_m = np.where(np.abs(ks - ks[i]) < delta_k)[0]
            valid_n = np.where(np.abs(ks - ks[j]) < delta_k)[0]
            sum = 0
            for m in valid_m:
                for n in valid_n:
                    sum += pks[m] * pks[n] * pk_cov[i, j] - pks[m] * pks[j] * pk_cov[i, n] - pks[i] * pks[n] * pk_cov[m, j] + pks[i] * pks[j] * pk_cov[m, n]
            cov[i, j] = prefactor * sum
    return np.array(cov)


if __name__ == "__main__":
    import seaborn as sb
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s |%(funcName)18s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    camb = getCambGenerator()
    r_s, _, _, _ = camb.get_data()
    extractor = PureBAOExtractor(r_s)
    mink = 0.02

    step_size = 1
    data_raw = PowerSpectrum_SDSS_DR12_Z061_NGC(step_size=step_size, min_k=0.0)
    data = PowerSpectrum_SDSS_DR12_Z061_NGC(step_size=step_size, min_k=mink)
    data2 = PowerSpectrum_SDSS_DR12_Z061_NGC(postprocess=extractor, step_size=step_size, min_k=mink)

    # Get all the data to compute the covariance
    ks = data_raw.ks
    pk = data_raw.data
    pk_cov = data_raw.cov
    denoms = extractor.postprocess(ks, pk, None, return_denominator=True)
    k_range = extractor.get_krange()

    cov_noda = calc_cov_noda(pk_cov, denoms, ks, pk, k_range, assume_diag=False)
    cov_noda2 = calc_cov_noda(pk_cov, denoms, ks, pk, k_range, assume_diag=True)
    cov_brute = data2.cov

    mask = np.where(ks < mink)
    cov_noda = np.delete(np.delete(cov_noda, mask, 0), mask, 1)
    cov_noda2 = np.delete(np.delete(cov_noda2, mask, 0), mask, 1)

    # Sanity check - print determinants
    print(np.linalg.det(cov_brute), np.linalg.det(cov_noda), np.linalg.det(cov_noda2))
    ii1 = cov_brute @ np.linalg.inv(cov_brute)
    ii2 = cov_noda @ np.linalg.inv(cov_noda)
    ii3 = cov_noda2 @ np.linalg.inv(cov_noda2)

    # A * inv(A) = I
    # If its not, there is an issue with the det being too close to zero
    print("This should all be one ", np.diag(ii1))
    print("This should all be one ", np.diag(ii2))
    print("This should all be one ", np.diag(ii3))

    icov_noda = np.linalg.inv(cov_noda2)
    icov_brute = np.linalg.inv(cov_brute)

    # Plot results
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
    axes = axes.flatten()
    axes[0].set_title("Covariance from Mocks")
    axes[1].set_title("Covariance from Nishimichi 2018, eq 7")
    axes[2].set_title("Precision mocks")
    axes[3].set_title("Precision Nishimichi")
    sb.heatmap(cov_brute, ax=axes[0])
    sb.heatmap(cov_noda2, ax=axes[1])
    sb.heatmap(icov_brute, ax=axes[2])
    sb.heatmap(icov_noda, ax=axes[3])
    plt.show()

    # The interesting finding here is that the covariance in P(k) significantly modifies
    # even the diagonal covariance in the BAO extractor method. Because the extractor utilises
    # a filter of pk values, correlation of pk enters directly into the diagonal error.
    # The paper versions of Noda report better constraints on alpha primarily because of this.
