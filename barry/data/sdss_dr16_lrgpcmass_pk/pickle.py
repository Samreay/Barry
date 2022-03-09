import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(
        loc,
        comment="#",
        skiprows=29,
        delim_whitespace=True,
        names=["k", "kmean", "pk0", "pk2", "pk4", "nk", "shot"],
    )
    df.loc[0, "k"] = 0.005  # The file has 0.005002, which causes a mismatch with the window function k.
    df["pk1"] = np.zeros(len(df))
    df["pk3"] = np.zeros(len(df))
    mask = df["nk"] >= 0
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
    return masked.astype(np.float32)


def getwin(winfile):
    # Hardcoded based on file's numbers
    kmin = 0.0
    kmax = 0.3
    kthmin = 0.0
    kthmax = 0.4
    nkth = 400
    nkout = 30
    step_size = 1

    files = [winfile]
    winfits = {}
    for winfit_file in files:
        print(f"Loading winfit from {winfit_file}")

        res = {}
        winfits[step_size] = res

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, skiprows=0)
        matrix = df.to_numpy().astype(np.float32)
        print(np.shape(matrix))

        new_matrix = np.zeros((5 * nkout, 5 * nkth))
        for ii, i in enumerate([0, 2, 4]):
            for jj, j in enumerate([0, 2, 4]):
                new_matrix[i * nkout : (i + 1) * nkout, j * nkth : (j + 1) * nkth] = matrix[
                    ii * nkout : (ii + 1) * nkout, jj * nkth : (jj + 1) * nkth
                ]

        winfits[step_size] = res
        res["w_ks_input"] = np.linspace(kthmin, kthmax, nkth, endpoint=False) + 0.5 * (kthmax - kthmin) / nkth
        res["w_k0_scale"] = np.zeros(nkth)  # Integral constraint is already included in matrix.
        res["w_transform"] = new_matrix
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
        return int(item.split("_")[11][:-4] if "postrecon" in item else int(item.split("_")[10][:-4]))


for gc in ["_NGC_", "_SGC_"]:
    ds = [
        f"/Volumes/Work/UQ/eBOSS_DR16/EZmocks_eBOSS_DR16_LRGpCMASS_prerecon/",
        f"/Volumes/Work/UQ/eBOSS_DR16/EZmocks_eBOSS_DR16_LRGpCMASS_postrecon/",
    ]
    files = [d + f for d in ds for f in os.listdir(d) if gc in f]
    files.sort(key=sortfunc)
    for file in files:
        print(file)

    datafiles = [
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_DR16LRG/lin/Power_Spectrum_comb{gc}dataNorm_datav72.txt",
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_postrec_DR16LRG/lin/Power_Spectrum_comb{gc}datav72_recon.txt",
    ]
    wfile = f"/Volumes/Work/UQ/eBOSS_DR16/Win_matrix_eBOSS_DR16_LRGpCMASS{gc}kmax_0.3_deltak_0p01_Mono+Quad+Hex.txt"
    res = {f.lower(): getdf(f) for f in files}
    datares = {f.lower(): getdf(f) for f in datafiles}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    print([v for k, v in res.items() if "postrecon" not in k])
    split = {
        "pre-recon data": [v for k, v in datares.items() if "postrec" not in k],
        "post-recon data": [v for k, v in datares.items() if "postrec" in k],
        "pre-recon mocks": [v for k, v in res.items() if "postrecon" not in k],
        "post-recon mocks": [v for k, v in res.items() if "postrecon" in k],
        "pre-recon cov": None,
        "post-recon cov": None,
        "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.698, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
        "name": f"eBOSS DR16 LRGpCMASS Pk {gc[1:-1]}",
        "winfit": getwin(wfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../sdss_dr16_lrgpcmass_pk{gc[:-1].lower()}.pkl", "wb") as f:
        pickle.dump(split, f)
