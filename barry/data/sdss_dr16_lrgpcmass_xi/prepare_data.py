import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getdata(xifile):
    df = pd.read_csv(xifile, comment="#", delim_whitespace=True, names=["s", "xi0", "err_xi0", "xi2", "err_xi2", "xi4", "err_xi4"]).drop(
        ["err_xi0", "err_xi2", "err_xi4"], axis=1
    )
    print(df)
    return df


def getcov(covfile, nss, recon=False):
    cov_input = pd.read_csv(covfile, comment="#", delim_whitespace=True, header=None).to_numpy()
    cov = np.zeros((3 * nss, 3 * nss))
    counter = 0
    index = 6 if recon else 4
    for l in range(3):
        for i in range(nss):
            for m in range(3):
                for j in range(nss):
                    cov[l * nss + i, m * nss + j] = cov_input[counter, index]
                    counter += 1
    plt.imshow(cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))
    plt.show()

    return cov


xi_pre = getdata("Data_LRGxi_NGCSGC_0.6z1.0_prerecon.txt")
xi_post = getdata("Data_LRGxi_NGCSGC_0.6z1.0_postrecon.txt")
covfile_pre = "Covariance_LRGxi_NGCSGC_0.6z1.0_prerecon.txt"
covfile_post = "Covariance_LRGxi_NGCSGC_0.6z1.0_postrecon.txt"

split = {
    "pre-recon data": [xi_pre],
    "post-recon data": [xi_post],
    "pre-recon mocks": None,
    "post-recon mocks": None,
    "pre-recon cov": getcov(covfile_pre, len(xi_pre["s"])),
    "post-recon cov": getcov(covfile_post, len(xi_post["s"]), recon=True),
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "mnu": 0.06, "reconsmoothscale": 15},
    "name": f"eBOSS DR16 LRGpCMASS Xi ngc+sgc",
}
with open("../sdss_dr16_lrgpcmass_xi_both.pkl", "wb") as f:
    pickle.dump(split, f)
