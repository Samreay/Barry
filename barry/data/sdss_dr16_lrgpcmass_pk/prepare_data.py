import pickle
import pandas as pd
import numpy as np
import os
from scipy.linalg import block_diag


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


def getdf_comb(loc):

    comb = []
    for l in loc:
        comb.append(getdf(l))
    comb = pd.concat(comb)
    return comb


def getwin(winfile):
    # Hardcoded based on file's numbers
    kmin = 0.0
    kmax = 0.3
    kthmin = 0.0
    kthmax = 0.4
    nkth = 400
    nkout = 30
    step_size = 1

    import matplotlib.pyplot as plt

    files = [winfile]
    winfits = {}
    for winfit_file in files:
        print(f"Loading winfit from {winfit_file}")

        res = {}
        winfits[step_size] = res

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, skiprows=0)
        matrix = df.to_numpy().astype(np.float32)

        new_matrix = np.zeros((5 * nkout, 6 * nkth))
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


def getwin_comb(winfiles):
    # Hardcoded based on file's numbers
    kmin = 0.0
    kmax = 0.3
    kthmin = 0.0
    kthmax = 0.4
    nkth = 400
    nkout = 30
    step_size = 1

    import matplotlib.pyplot as plt

    winfits = {}
    matrices = []
    for winfit_file in winfiles:
        print(f"Loading winfit from {winfit_file}")

        df = pd.read_csv(winfit_file, comment="#", delim_whitespace=True, skiprows=0)
        matrix = df.to_numpy().astype(np.float32)
        new_matrix = np.zeros((5 * nkout, 6 * nkth))
        for ii, i in enumerate([0, 2, 4]):
            for jj, j in enumerate([0, 2, 4]):
                new_matrix[i * nkout : (i + 1) * nkout, j * nkth : (j + 1) * nkth] = matrix[
                    ii * nkout : (ii + 1) * nkout, jj * nkth : (jj + 1) * nkth
                ]
        matrices.append(new_matrix)

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
    res["w_ks_input"] = np.linspace(kthmin, kthmax, nkth, endpoint=False) + 0.5 * (kthmax - kthmin) / nkth
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
        if "prerecon" in item:
            if "NGC" in item:
                base = 0
            else:
                base = 1
        else:
            if "NGC" in item:
                base = 2
            else:
                base = 3
        val = int(item.split("_")[11][:-4] if "postrecon" in item else int(item.split("_")[10][:-4]))
        return base + 4 * val


ds = [
    f"/Volumes/Work/UQ/eBOSS_DR16/EZmocks_eBOSS_DR16_LRGpCMASS_prerecon/",
    f"/Volumes/Work/UQ/eBOSS_DR16/EZmocks_eBOSS_DR16_LRGpCMASS_postrecon/",
]
print(ds)
files = [d + f for d in ds for f in os.listdir(d) if "_NGCSGC_" not in f]
files.sort(key=sortfunc)
# for file in files:
#    print(file)

# Separate NGC and SGC
res = {f.lower(): getdf(f) for f in files}
ks = next(iter(res.items()))[1]["k"].to_numpy()

for gc in ["_NGC_", "_SGC_"]:
    wfile = f"/Volumes/Work/UQ/eBOSS_DR16/Win_matrix_eBOSS_DR16_LRGpCMASS{gc}kmax_0.3_deltak_0p01_Mono+Quad+Hex.txt"
    datafiles = [
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_DR16LRG/lin/Power_Spectrum_comb{gc}dataNorm_datav72.txt",
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_postrec_DR16LRG/lin/Power_Spectrum_comb{gc}datav72_recon.txt",
    ]
    datares = {f.lower(): getdf(f) for f in datafiles}
    split = {
        "n_data": 1,
        "pre-recon data": [v for k, v in datares.items() if "postrec" not in k],
        "post-recon data": [v for k, v in datares.items() if "postrec" in k],
        "pre-recon mocks": [v for k, v in res.items() if "postrecon" not in k and gc.lower() in k],
        "post-recon mocks": [v for k, v in res.items() if "postrecon" in k and gc.lower() in k],
        "pre-recon cov": None,
        "post-recon cov": None,
        "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.698, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
        "name": f"eBOSS DR16 LRGpCMASS Pk {gc[1:-1]}",
        "winfit": getwin(wfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }
    print(len(split["pre-recon data"]), len(split["post-recon data"]), len(split["pre-recon mocks"]), len(split["post-recon mocks"]))

    with open(f"../sdss_dr16_lrgpcmass_pk{gc[:-1].lower()}.pkl", "wb") as f:
        pickle.dump(split, f)

# Combined NGC and SGC by pairing up the files from the same data/mock
files = [files[i : i + 2] for i in range(0, len(files), 2)]
# for file in files:
#    print(file)
res = {f[0].replace("_NGC_", "_").lower(): getdf_comb(f) for f in files}
ks = next(iter(res.items()))[1]["k"].to_numpy()

wfiles = [
    f"/Volumes/Work/UQ/eBOSS_DR16/Win_matrix_eBOSS_DR16_LRGpCMASS{gc}kmax_0.3_deltak_0p01_Mono+Quad+Hex.txt" for gc in ["_NGC_", "_SGC_"]
]
datafiles = [
    [
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_DR16LRG/lin/Power_Spectrum_comb_NGC_dataNorm_datav72.txt",
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_DR16LRG/lin/Power_Spectrum_comb_SGC_dataNorm_datav72.txt",
    ],
    [
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_postrec_DR16LRG/lin/Power_Spectrum_comb_NGC_datav72_recon.txt",
        f"/Volumes/Work/UQ/eBOSS_DR16/Power_Spectrum_data_postrec_DR16LRG/lin/Power_Spectrum_comb_SGC_datav72_recon.txt",
    ],
]
datares = {f[0].replace("_NGC_", "_"): getdf_comb(f) for f in datafiles}
split = {
    "n_data": 2,
    "pre-recon data": [v for k, v in datares.items() if "postrec" not in k],
    "post-recon data": [v for k, v in datares.items() if "postrec" in k],
    "pre-recon mocks": [v for k, v in res.items() if "postrecon" not in k],
    "post-recon mocks": [v for k, v in res.items() if "postrecon" in k],
    "pre-recon cov": None,
    "post-recon cov": None,
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.698, "ob": 0.04814, "ns": 0.97, "mnu": 0.0, "reconsmoothscale": 15},
    "name": f"eBOSS DR16 LRGpCMASS Pk ngc+sgc",
    "winfit": getwin_comb(wfiles),
    "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
    "m_mat": getcomp_comb(2),
}

print(len(split["pre-recon data"]), len(split["post-recon data"]), len(split["pre-recon mocks"]), len(split["post-recon mocks"]))

with open(f"../sdss_dr16_lrgpcmass_pk_both.pkl", "wb") as f:
    pickle.dump(split, f)
