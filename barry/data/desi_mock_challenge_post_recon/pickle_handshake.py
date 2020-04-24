import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(loc, comment="#", skiprows=10, delim_whitespace=True, names=["k", "keff", "pk0", "pk2", "pk4", "nk", "shot"])
    mask = df["nk"] >= 0
    masked = df.loc[mask, ["k", "pk0", "pk2", "pk4", "nk"]]
    return masked.astype(np.float32)


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(3 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.eye(3 * ks.size)
    return matrix


if __name__ == "__main__":

    ds = f"/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/EZmocks/pk_EZmocks/"
    files = [ds + f for f in os.listdir(ds) if "Power_Spectrum" in f and "RSD" in f]
    print(files)

    pk_filename = f"/global/project/projectdirs/desi/users/UNIT-BAO-RSD-challenge/UNIT/pk/Power_Spectrum_UNIT_HODsnap97_ELGv1_redshift.txt"
    cov_filename = ds + f"/covariance/cov_matrix_pk-EZmocks_rsd.txt"
    data = getdf(pk_filename)
    ks = data["k"].to_numpy()
    cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    cov = (cov.astype(np.float32)[:, 2]).reshape((3 * len(ks), 3 * len(ks)))

    res = {f.lower(): getdf(f) for f in files}
    split = {
        "pre-recon data": [data],
        "pre-recon cov": cov,
        "post-recon data": None,
        "post-recon cov": None,
        "pre-recon mocks": [v for k, v in res.items()],
        "post-recon mocks": None,
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
