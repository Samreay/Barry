import logging

import numpy as np
from barry.cosmology.camb_generator import getCambGenerator
from barry.datasets import PowerSpectrum_SDSS_DR12_Z061_NGC
from barry.postprocessing import BAOExtractor, PureBAOExtractor


def get_rs_rs_index(pk_cov, denoms, ks, pks, delta_k, i, j):
    prefactor = 1 / (denoms[i] * denoms[j] * (pks[i] * pks[j]) ** 2)

    # Here is our first issue. The paper 1901.06854 does not define m or n,
    # however, from 1708.00375v3 we have that it is the values within k_range
    valid_m = np.where(np.abs(ks - ks[i]) < delta_k)[0]
    valid_n = np.where(np.abs(ks - ks[j]) < delta_k)[0]
    sum = 0
    for m in valid_m:
        for n in valid_n:
            sum += pks[m] * pks[n] * pk_cov[i, j] - pks[m] * pks[j] * pk_cov[i, n] - pks[i] * pks[n] * pk_cov[m, j] + pks[i] * pks[j] * pk_cov[m, n]
    return sum * prefactor


def get_rs_ps_index(pk_cov, denoms, ks, pks, delta_k, i, j):
    prefactor = 1 / (denoms[i] * (pks[i] ** 2))
    valid_l = np.where(np.abs(ks - ks[i]) < delta_k)[0]
    sum = 0
    for l in valid_l:
        sum += pks[l] * pk_cov[i, j] - pks[i] * pk_cov[l, j]
    return sum * prefactor


def calc_cov_noda_mixed(pk_cov, denoms, ks, pks, delta_k, is_extracted):
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

    logging.basicConfig(level=logging.INFO, format="[%(levelname)7s |%(funcName)18s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    camb = getCambGenerator()
    r_s, _ = camb.get_data()
    extractor = BAOExtractor(r_s, reorder=False)
    extractor2 = PureBAOExtractor(r_s)

    step_size = 1
    mink = 0.02
    maxk = 0.3
    data_raw = PowerSpectrum_SDSS_DR12_Z061_NGC(step_size=step_size, fake_diag=False, min_k=0.0, max_k=0.32)
    data2 = PowerSpectrum_SDSS_DR12_Z061_NGC(postprocess=extractor, step_size=step_size, min_k=mink, max_k=maxk)

    ks = data_raw.ks
    pk = data_raw.data
    pk_cov = data_raw.cov
    denoms = extractor2.postprocess(ks, pk, None, return_denominator=True)
    is_extracted = extractor.get_is_extracted(ks)
    cov_brute = data2.cov
    k_range = extractor.get_krange()
    cov_noda = calc_cov_noda_mixed(pk_cov, denoms, ks, pk, k_range, is_extracted)
    cov_noda_diag = calc_cov_noda_mixed(np.diag(np.diag(pk_cov)), denoms, ks, pk, k_range, is_extracted)

    mask = (ks >= mink) & (ks <= maxk)
    mask2d = mask[np.newaxis, :] & mask[:, np.newaxis]

    cov_noda = cov_noda[mask2d].reshape((mask.sum(), mask.sum()))
    cov_noda_diag = cov_noda_diag[mask2d].reshape((mask.sum(), mask.sum()))

    print("Raw pk should be ones: ", np.diag(pk_cov @ np.linalg.inv(pk_cov)))
    print("Brute mix should be ones: ", np.diag(cov_brute @ np.linalg.inv(cov_brute)))
    print("Noda mix should be ones: ", np.diag(cov_noda @ np.linalg.inv(cov_noda)))
    print("Noda mix diag should be ones: ", np.diag(cov_noda_diag @ np.linalg.inv(cov_noda_diag)))

    la_cov_brute = np.log(np.abs(cov_brute))
    la_cov_noda = np.log(np.abs(cov_noda) + np.abs(cov_brute).min())
    la_cov_noda_diag = np.log(np.abs(cov_noda_diag) + np.abs(cov_brute).min())

    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    axes[0].set_title("Mocks")
    axes[1].set_title("Nishimichi 2018, (Eq7)")
    axes[2].set_title("Nishimichi 2018, (Eq7), diagonal")
    # axes[3].set_title("norm diff first two, capped at unity")
    vmin = min(la_cov_brute.min(), la_cov_noda.min(), la_cov_noda_diag.min())
    vmax = max(la_cov_brute.max(), la_cov_noda.max(), la_cov_noda_diag.max())
    sb.heatmap(la_cov_brute, ax=axes[0], vmin=vmin, vmax=vmax)
    sb.heatmap(la_cov_noda, ax=axes[1], vmin=vmin, vmax=vmax)
    sb.heatmap(la_cov_noda_diag, ax=axes[2], vmin=vmin, vmax=vmax)
    # sb.heatmap((cov_brute - cov_noda_diag) / cov_brute, ax=axes[3], vmin=-1, vmax=1)
    fig.subplots_adjust(hspace=0.0)

    output = "plots"
    import os

    filename = os.path.join(output, "covariance_mixed.png")
    os.makedirs(output, exist_ok=True)
    plt.show()
    fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")

    # FINDINGS
    # Their math is good... but not their assumption you can use diagonal covariance for the underlying
    # pk uncertainty.
