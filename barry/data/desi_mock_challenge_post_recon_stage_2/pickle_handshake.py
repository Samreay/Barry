import pickle
import pandas as pd
import numpy as np
import os


def getxi(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["s", "xi0", "xi2", "xi4"], header=None)
    mask = df["s"] <= 205.0
    masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
    print(masked)
    return masked.astype(np.float32)


def getpk(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["k", "pk0", "pk2", "pk4"], header=None)
    df["pk1"] = 0
    df["pk3"] = 0
    mask = df["k"] <= 0.5
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4"]]
    return masked.astype(np.float32)


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(5 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.zeros((5 * ks.size, 3 * ks.size))
    matrix[: ks.size, : ks.size] = np.diag(np.ones(ks.size))
    matrix[2 * ks.size : 3 * ks.size, ks.size : 2 * ks.size] = np.diag(np.ones(ks.size))
    matrix[4 * ks.size :, 2 * ks.size :] = np.diag(np.ones(ks.size))
    return matrix


def format_pk_cov(nks, covfile):

    cov = pd.read_csv(covfile, delim_whitespace=True, header=None).to_numpy()
    cov_flat = cov.astype(np.float64)[:, 2]
    nin = int(np.sqrt(len(cov)) / 3)
    cov_input = cov_flat.reshape((3 * nin, 3 * nin))
    cov = np.zeros((5 * nks, 5 * nks))
    cov[:nks, :nks] = cov_input[:nks, :nks]
    cov[:nks, 2 * nks : 3 * nks] = cov_input[:nks, nin : nin + nks]
    cov[:nks, 4 * nks :] = cov_input[:nks, 2 * nin : 2 * nin + nks]
    cov[2 * nks : 3 * nks, :nks] = cov_input[nin : nin + nks, :nks]
    cov[2 * nks : 3 * nks, 2 * nks : 3 * nks] = cov_input[nin : nin + nks, nin : nin + nks]
    cov[2 * nks : 3 * nks, 4 * nks :] = cov_input[nin : nin + nks, 2 * nin : 2 * nin + nks]
    cov[4 * nks :, :nks] = cov_input[2 * nin : 2 * nin + nks, :nks]
    cov[4 * nks :, 2 * nks : 3 * nks] = cov_input[2 * nin : 2 * nin + nks, nin : nin + nks]
    cov[4 * nks :, 4 * nks :] = cov_input[2 * nin : 2 * nin + nks, 2 * nin : 2 * nin + nks]

    return cov


def format_xi_cov(nss, covfile):

    cov = pd.read_csv(covfile, delim_whitespace=True, header=None).to_numpy()
    cov_flat = cov.astype(np.float32)[:, 2]
    nin = int(np.sqrt(len(cov)) / 3)
    cov_input = cov_flat.reshape((3 * nin, 3 * nin))
    cov = np.zeros((3 * nss, 3 * nss))
    cov[:nss, :nss] = cov_input[start : start + nss, start : start + nss]
    cov[:nss, nss : 2 * nss] = cov_input[start : start + nss, start + nin : start + nin + nss]
    cov[:nss, 2 * nss :] = cov_input[start : start + nss, start + 2 * nin : start + 2 * nin + nss]
    cov[nss : 2 * nss, :nss] = cov_input[start + nin : start + nin + nss, start : start + nss]
    cov[nss : 2 * nss, nss : 2 * nss] = cov_input[start + nin : start + nin + nss, start + nin : start + nin + nss]
    cov[nss : 2 * nss, 2 * nss :] = cov_input[start + nin : start + nin + nss, start + 2 * nin : start + 2 * nin + nss]
    cov[2 * nss :, :nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start : start + nss]
    cov[2 * nss :, nss : 2 * nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + nin : start + nin + nss]
    cov[2 * nss :, 2 * nss :] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + 2 * nin : start + 2 * nin + nss]

    return cov


def collect_pk_data(pre_files, post_files, pre_covfile, post_covfile, a, smooth, fix):

    print(pre_files)
    print(post_files)
    print(pre_covfile)
    print(post_covfile)

    pre_res = {f.lower(): getpk(f) for f in pre_files}
    post_res = {f.lower(): getpk(f) for f in pre_files}

    ks = next(iter(pre_res.items()))[1]["k"].to_numpy()
    pre_cov = format_pk_cov(len(ks), pre_covfile)
    print(np.shape(pre_cov), np.shape(ks))

    ks = next(iter(pre_res.items()))[1]["k"].to_numpy()
    post_cov = format_pk_cov(len(ks), post_covfile)
    print(np.shape(post_cov), np.shape(ks))

    split = {
        "pre-recon data": [v for k, v in pre_res.items()],
        "pre-recon cov": pre_cov,
        "post-recon data": [v for k, v in post_res.items()],
        "post-recon cov": post_cov,
        "pre-recon mocks": None,
        "post-recon mocks": None,
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": smooth,
        },
        "name": f"DESI Mock Challenge Stage 2 Pk " + a + " " + str(smooth) + " " + fix,
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../desi_mock_challenge_post_stage_2_pk_" + a + "_" + str(smooth) + fix + ".pkl", "wb") as f:
        pickle.dump(split, f)


def collect_xi_data(pre_files, post_files, pre_covfile, post_covfile, a, smooth, fix):

    print(pre_files)
    print(post_files)
    print(pre_covfile)
    print(post_covfile)

    pre_res = {f.lower(): getxi(f) for f in pre_files}
    post_res = {f.lower(): getxi(f) for f in pre_files}

    nss = len(next(iter(pre_res.items()))[1]["s"].to_numpy())
    pre_cov = format_xi_cov(nss, pre_covfile)
    print(np.shape(pre_cov), nss)

    ks = len(next(iter(pre_res.items()))[1]["k"].to_numpy())
    post_cov = format_pk_cov(len(ks), post_covfile)
    print(np.shape(post_cov), nss)

    split = {
        "pre-recon data": [v for k, v in pre_res.items()],
        "pre-recon cov": pre_cov,
        "post-recon data": [v for k, v in post_res.items()],
        "post-recon cov": post_cov,
        "pre-recon mocks": None,
        "post-recon mocks": None,
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": smooth,
        },
        "name": f"DESI Mock Challenge Stage 2 Xi" + a + " " + str(smooth) + " " + fix,
    }

    with open(f"../desi_mock_challenge_post_stage_2_xi_" + a + "_" + str(smooth) + fix + ".pkl", "wb") as f:
        pickle.dump(split, f)


if __name__ == "__main__":

    # =======
    # Non-Fix
    ds = f"/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/Reconstruction/Stage2_3Gpc/Multipoles/"
    covds = f"/global/project/projectdirs/desi/users/jmen_a/EZmocks/EZmocks_1Gpc_recons_nonfix/"

    # Power Spectrum
    pre_files = [
        "UNIELG-b0s05rsd0g1536premultipoles",
        "UNIELG-b0s10rsd0g1536premultipoles",
        "UNIELG-b0s15rsd0g1536premultipoles",
    ]
    post_files = [
        "UNIELG-b0s05rsd0g1536postmultipoles",
        "UNIELG-b0s10rsd0g1536postmultipoles",
        "UNIELG-b0s15rsd0g1536postmultipoles",
    ]
    for j, (a, b, c) in enumerate(zip(["iso", "ani"], ["Yuyu_RecIso", "Yuyu_RecAni"], ["RecIsoNonfix", "RecAniNonfix"])):
        for i, smooth in enumerate([5, 10, 15]):
            pre_file = [ds + b + "/dk0.005kmin0.005/" + pre_files[i] + ".txt"]
            post_file = [ds + b + "/dk0.005kmin0.005/" + post_files[i] + ".txt"]
            pre_covfile = covds + c + "cov_matrix_pk-EZmocks-1Gpc-" + c[:6] + str("Sm%d" % i) + "-nonfix_rsd_pre.txt"
            post_covfile = covds + c + "cov_matrix_pk-EZmocks-1Gpc-" + c[:6] + str("Sm%d" % i) + "-nonfix_rsd_post.txt"
            collect_pk_data(pre_file, post_file, pre_covfile, post_covfile, a, smooth, "nonfix")

            """pre_file = [
                "/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/UNIT/xi_3Gpc_v2/2PCF_UNIT_DESI_Shadab_HOD_snap97_ELG_v1_xil.dat"
            ]
            post_file = [ds + b + "/dk0.005kmin0.005/" + post_files[i] + "_xi_gs_han4.txt"]
            pre_covfile = "/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/EZmocks/xi_EZmocks_nonfix/covariance_3Gpc_v2/cov_matrix_xi-EZmocks-3Gpc-nonfix_rsd_centerbin.txt"
            post_covfile = covds + c + "cov_matrix_pk-EZmocks-1Gpc-" + c[:6] + str("Sm%d" % i) + "-nonfix_rsd_post.txt"
            collect_xi_data(pre_file, post_file, pre_covfile, post_covfile, a, smooth, "nonfix")"""
