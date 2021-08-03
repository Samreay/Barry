import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    skip = 34 if "recon" in loc else 31
    if "data" in loc:
        df = pd.read_csv(
            loc,
            comment="#",
            skiprows=skip,
            delim_whitespace=True,
            names=["k", "kmean", "pk0", "pk2", "nk"],
        )
        df["pk4"] = np.zeros(len(df))
    else:
        df = pd.read_csv(
            loc,
            comment="#",
            skiprows=skip,
            delim_whitespace=True,
            names=["k", "kmean", "pk0", "pk2", "pk4", "sigma_pk_lin", "nk"],
        )
    df["pk1"] = np.zeros(len(df))
    df["pk3"] = np.zeros(len(df))
    mask = df["nk"] >= 0
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
    return masked.astype(np.float32)


def getwin(winfile):
    # Hardcoded based on Florian's numbers
    kmin = 0.0
    kmax = 0.4
    nkth = 400
    nkout = 40
    step_size = 2

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


def getcomp():
    nkth = 400
    matrix = np.zeros((5 * nkth, 3 * nkth))
    matrix[:nkth, :nkth] = np.diag(np.ones(nkth))
    matrix[2 * nkth : 3 * nkth, nkth : 2 * nkth] = np.diag(np.ones(nkth))
    matrix[4 * nkth :, 2 * nkth :] = np.diag(np.ones(nkth))
    print(np.shape(matrix))
    return matrix


def sortfunc(item):
    if "data" in item:
        return 0
    else:
        return int(item.split("_")[11] if "recon" in item else item.split("_")[9])


for gc in ["ngc_z3", "sgc_z3"]:
    ds = [f"/Volumes/Work/UQ/DR12/ps1d_patchyDR12_{gc}/", f"/Volumes/Work/UQ/DR12/ps1d_patchyDR12_{gc}_recon/"]
    files = [d + f for d in ds for f in os.listdir(d)]
    files.sort(key=sortfunc)
    for file in files:
        print(file)

    wfile = f"/Volumes/Work/UQ/DR12/Wll_{gc}_rebinned_5000bins_s10fixed.dat"
    res = {f.lower(): getdf(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    split = {
        "pre-recon data": [v for k, v in res.items() if "recon" not in k and "data" in k],
        "post-recon data": [v for k, v in res.items() if "recon" in k and "data" in k],
        "pre-recon mocks": [v for k, v in res.items() if "recon" not in k and "data" not in k],
        "post-recon mocks": [v for k, v in res.items() if "recon" in k and "data" not in k],
        "pre-recon cov": None,
        "post-recon cov": None,
        "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
        "name": f"BOSS DR12 Pk {gc}",
        "winfit": getwin(wfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../sdss_dr12_pk_{gc.lower()}.pkl", "wb") as f:
        pickle.dump(split, f)
