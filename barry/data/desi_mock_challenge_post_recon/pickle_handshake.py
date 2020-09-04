import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["k", "pk0", "pk2"])
    df["pk1"] = 0
    df["pk3"] = 0
    df["pk4"] = 0
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


if __name__ == "__main__":

    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO/"
    files = [ds + f for f in os.listdir(ds) if "pkl" in f]
    print(files)

    cov_filename = ds + f"/cov_matrix_pk-EZmocks-1Gpc_rsd_centerbin_post.txt"
    res = {f.lower(): getdf(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    cov_flat = cov.astype(np.float32)[:, 2]
    nin = int(np.sqrt(len(cov)) / 3)
    cov_input = cov_flat.reshape((3 * nin, 3 * nin))
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = cov_input[: len(ks), : len(ks)]
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[: len(ks), nin : nin + len(ks)]
    cov[: len(ks), 4 * len(ks) :] = cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = cov_input[nin : nin + len(ks), : len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[nin : nin + len(ks), nin : nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[4 * len(ks) :, : len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]
    cov[4 * len(ks) :, 4 * len(ks) :] = cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]

    print(np.shape(cov), np.shape(ks))
    print([k for k, v in res.items()])

    split = {
        "pre-recon data": None,
        "pre-recon cov": None,
        "post-recon data": None,
        "post-recon cov": cov,
        "pre-recon mocks": None,
        "post-recon mocks": [v for k, v in res.items()],
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Mock Challenge Handshake Pk",
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../desi_mock_challenge_handshake.pkl", "wb") as f:
        pickle.dump(split, f)
