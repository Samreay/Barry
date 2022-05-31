import pickle
import pandas as pd
import numpy as np
import os
from scipy.linalg import block_diag


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
    if len(df) == 0:
        print(loc)
    df["pk1"] = np.zeros(len(df))
    df["pk3"] = np.zeros(len(df))
    mask = df["nk"] >= 0
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
    return masked.astype(np.float32)


def getdf_comb(loc):

    comb = []
    for l in loc:
        skip = 34 if "recon" in l else 31
        if "data" in l:
            df = pd.read_csv(
                l,
                comment="#",
                skiprows=skip,
                delim_whitespace=True,
                names=["k", "kmean", "pk0", "pk2", "nk"],
            )
            df["pk4"] = np.zeros(len(df))
        else:
            df = pd.read_csv(
                l,
                comment="#",
                skiprows=skip,
                delim_whitespace=True,
                names=["k", "kmean", "pk0", "pk2", "pk4", "sigma_pk_lin", "nk"],
            )
        if len(df) == 0:
            print(l)
        df["pk1"] = np.zeros(len(df))
        df["pk3"] = np.zeros(len(df))
        mask = df["nk"] >= 0
        masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
        comb.append(masked.astype(np.float32))
    comb = pd.concat(comb)
    return comb


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
        matrix = np.eye(5 * nkout, 6 * nkth)
        print(f"Loading winfit from {winfit_file}")

        res = {}
        winfits[step_size] = res

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, header=None)
        matrix[: 5 * nkout, : 5 * nkth] = df.to_numpy().astype(np.float32)

        winfits[step_size] = res
        res["w_ks_input"] = np.linspace(kmin, kmax, nkth, endpoint=False) + 0.5 * (kmax - kmin) / nkth
        res["w_k0_scale"] = np.zeros(nkth)  # Integral constraint is already included in matrix.
        res["w_transform"] = matrix
        res["w_ks_output"] = np.linspace(kmin, kmax, nkout, endpoint=False) + 0.5 * (kmax - kmin) / nkout
    return winfits


def getwin_comb(winfiles):
    # Hardcoded based on Florian's numbers
    kmin = 0.0
    kmax = 0.4
    nkth = 400
    nkout = 40
    step_size = 2

    winfits = {}
    matrices = []
    for winfit_file in winfiles:
        matrix = np.eye(5 * nkout, 6 * nkth)
        print(f"Loading winfit from {winfit_file}")

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, header=None)
        matrix[: 5 * nkout, : 5 * nkth] = df.to_numpy().astype(np.float32)
        matrices.append(matrix)
        print(np.shape(matrix))

    nw = len(winfiles)
    matrix = np.zeros((nw * 5 * nkout, nw * 6 * nkth))
    for i in range(5):
        for j in range(5):
            minth, maxth = i * nkth, (i + 1) * nkth
            minout, maxout = j * nkout, (j + 1) * nkout
            matrix[nw * minout : nw * maxout, nw * minth : nw * maxth] = block_diag(
                *[matrices[d][minout:maxout, minth:maxth] for d in range(nw)]
            )

    winfits[step_size] = res
    res["w_ks_input"] = np.linspace(kmin, kmax, nkth, endpoint=False) + 0.5 * (kmax - kmin) / nkth
    res["w_k0_scale"] = np.tile(np.zeros(nkth), 2)  # Integral constraint is already included in matrix.
    res["w_transform"] = matrix
    res["w_ks_output"] = np.linspace(kmin, kmax, nkout, endpoint=False) + 0.5 * (kmax - kmin) / nkout
    return winfits


def getcomp():
    nkth = 400
    matrix = np.zeros((6 * nkth, 3 * nkth))
    matrix[:nkth, :nkth] = np.diag(np.ones(nkth))
    matrix[2 * nkth : 3 * nkth, nkth : 2 * nkth] = np.diag(np.ones(nkth))
    matrix[4 * nkth : 5 * nkth, 2 * nkth :] = np.diag(np.ones(nkth))
    return matrix


def getcomp_comb(ncomb):

    nkth = 400
    matrix = np.zeros((ncomb * 6 * nkth, ncomb * 3 * nkth))
    matrix[: ncomb * nkth, : ncomb * nkth] = np.diag(np.ones(ncomb * nkth))
    matrix[2 * ncomb * nkth : 3 * ncomb * nkth, ncomb * nkth : 2 * ncomb * nkth] = np.diag(np.ones(ncomb * nkth))
    matrix[4 * ncomb * nkth : 5 * ncomb * nkth, 2 * ncomb * nkth :] = np.diag(np.ones(ncomb * nkth))
    return matrix


def sortfunc(item):
    if "data" in item:
        return 0
    else:
        return int(item.split("_")[11] if "recon" in item else item.split("_")[9])


for z, (zname, red) in enumerate(zip(["z1", "z2", "z3"], [0.38, 0.51, 0.61])):

    # Seperate files for NGC and SGC
    ds = [
        f"/Volumes/Work/UQ/DR12/ps1d_patchyDR12_{gc}/"
        for gc in [f"ngc_{zname}", f"sgc_{zname}", f"ngc_{zname}_recon", f"sgc_{zname}_recon"]
    ]
    print(ds)
    files = [d + f for d in ds for f in os.listdir(d)]
    files.sort(key=sortfunc)
    for file in files:
        print(file)

    # Separate NGC and SGC
    res = {f.lower(): getdf(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()

    for i, gc in enumerate([f"ngc_{zname}", f"sgc_{zname}"]):
        wfile = f"/Volumes/Work/UQ/DR12/Wll_{gc}_rebinned_5000bins_s10fixed.dat"
        split = {
            "n_data": 1,
            "pre-recon data": [v for k, v in res.items() if "recon" not in k and "data" in k and gc[:3] in k],
            "post-recon data": [v for k, v in res.items() if "recon" in k and "data" in k and gc[:3] in k],
            "pre-recon mocks": [v for k, v in res.items() if "recon" not in k and "data" not in k and gc[:3] in k],
            "post-recon mocks": [v for k, v in res.items() if "recon" in k and "data" not in k and gc[:3] in k],
            "pre-recon cov": None,
            "post-recon cov": None,
            "cosmology": {"om": 0.31, "h0": 0.676, "z": red, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
            "name": f"BOSS DR12 Pk {gc}",
            "winfit": getwin(wfile),
            "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
            "m_mat": getcomp(),
        }
        print(len(split["pre-recon data"]), len(split["post-recon data"]), len(split["pre-recon mocks"]), len(split["post-recon mocks"]))

        with open(f"../sdss_dr12_pk_{gc.lower()}.pkl", "wb") as f:
            pickle.dump(split, f)

    # Combined NGC and SGC by pairing up the files from the same data/mock
    files = [files[i : i + 2] for i in range(0, len(files), 2)]
    res = {f[0].replace("ngc", "").lower(): getdf_comb(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()

    wfiles = [f"/Volumes/Work/UQ/DR12/Wll_{gc}_rebinned_5000bins_s10fixed.dat" for gc in [f"ngc_{zname}", f"sgc_{zname}"]]
    split = {
        "n_data": 2,
        "pre-recon data": [v for k, v in res.items() if "recon" not in k and "data" in k],
        "post-recon data": [v for k, v in res.items() if "recon" in k and "data" in k],
        "pre-recon mocks": [v for k, v in res.items() if "recon" not in k and "data" not in k],
        "post-recon mocks": [v for k, v in res.items() if "recon" in k and "data" not in k],
        "pre-recon cov": None,
        "post-recon cov": None,
        "cosmology": {"om": 0.31, "h0": 0.676, "z": red, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
        "name": f"BOSS DR12 Pk ngc+sgc_{zname}",
        "winfit": getwin_comb(wfiles),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp_comb(2),
    }

    with open(f"../sdss_dr12_pk_both_{zname}.pkl", "wb") as f:
        pickle.dump(split, f)
