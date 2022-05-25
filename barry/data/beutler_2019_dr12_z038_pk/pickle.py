import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(
        loc,
        comment="#",
        skiprows=31,
        delim_whitespace=True,
        names=["k", "kmean", "pk0", "sigma_pk0", "pk1", "sigma_pk1", "pk2", "sigma_pk2", "pk3", "sigma_pk3", "pk4", "sigma_pk4", "nk"],
    )
    mask = df["nk"] >= 0
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
    return masked.astype(np.float32)


def getwin(winfile):
    # Hardcoded based on Florian's numbers
    kmin = 0.0
    kmax = 0.4
    nkth = 400
    nkout = 40
    step_size = 10

    files = [winfile]
    winfits = {}
    for winfit_file in files:
        print(f"Loading winfit from {winfit_file}")

        res = {}
        winfits[step_size] = res

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, header=None)
        matrix = df.to_numpy().astype(np.float32)
        print(np.shape(matrix))

        winfits[step_size] = res
        res["w_ks_input"] = np.linspace(kmin, kmax, nkth, endpoint=False) + 0.5 * (kmax - kmin) / nkth
        res["w_k0_scale"] = np.zeros(nkth)  # Integral constraint is already included in matrix.
        res["w_transform"] = matrix
        res["w_ks_output"] = np.linspace(kmin, kmax, nkout, endpoint=False) + 0.5 * (kmax - kmin) / nkout
    return winfits


def getcomp(compfile):
    print(f"Loading compression file from {compfile}")
    df = pd.read_csv(compfile, comment="#", delim_whitespace=True, header=None)
    matrix = df.to_numpy().astype(np.float32)
    print(np.shape(matrix))
    return matrix


if __name__ == "__main__":

    for gc in ["NGC", "SGC"]:
        mfile = f"M2D_pk_BOSS_DR12_{gc}_z1_1_1_1_1_1_10_10.dat"
        wfile = f"W2D_pk_BOSS_DR12_{gc}_z1_1_1_1_1_1_10_10.dat"
        ds = [f"pre_recon_{gc}/"]  # Don't have post-recon files
        files = [d + f for d in ds for f in os.listdir(d)]
        print(files)

        res = {f.lower(): getdf(f) for f in files}
        split = {
            "n_data": 1,
            "pre-recon data": [v for k, v in res.items() if "pre_recon" in k and "patchy" not in k],
            "post-recon data": None,
            "pre-recon mocks": [v for k, v in res.items() if "pre_recon" in k and "patchy" in k],
            "post-recon mocks": None,
            "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.38, "ob": 0.04814, "ns": 0.97, "mnu": 0.00, "reconsmoothscale": 15},
            "name": f"Beutler 2019 Z0.38 Pk {gc}",
            "winfit": getwin(wfile),
            "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
            "m_mat": getcomp(mfile),
        }

        with open(f"../beutler_2019_dr12_z038_pk_{gc.lower()}.pkl", "wb") as f:
            pickle.dump(split, f)
