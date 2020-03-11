import pickle
import pandas as pd
import numpy as np


def getdf(loc):
    df = pd.read_csv(loc, comment="#", delim_whitespace=True, names=["k", "pk0", "pk2"])
    return df.astype(np.float32)


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(2 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.eye(2 * ks.size)
    return matrix


if __name__ == "__main__":
    pk_filename = "Pk_multipoles_BAO_fitting_DC.v0.dat"
    cov_filename = "Pk_multipoles_cov_BAO_fitting_DC.v0.dat"

    data = getdf(pk_filename)
    ks = data["k"].to_numpy()
    cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    split = {
        "pre-recon data": [data],
        "pre-recon cov": cov.astype(np.float32),
        "post-recon data": None,
        "post-recon cov": None,
        "pre-recon mocks": None,
        "post-recon mocks": None,
        "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "reconsmoothscale": 15},
        "name": f"DESI Mock BAO Challenge 0, z=   Pk",
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../desi_mock_challenge_0.pkl", "wb") as f:
        pickle.dump(split, f)
