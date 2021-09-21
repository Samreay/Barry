import pickle
import pandas as pd
import numpy as np
import os


def getxi(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["s", "xi0", "xi2", "xi4"], header=None)
    print(df)
    mask = df["s"] <= 205.0
    masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
    return masked.astype(np.float32)


def getpk(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["k", "pk0", "pk2", "pk4", "nk"], header=None)
    df["k"] = (
        np.linspace(0.0, 0.48, 48, endpoint=False) + 0.01 / 2.0
    )  # Overwrite the k_mean values to k_central, which is appropriate as we are using a window function where the k-bin has been integrated over.
    df["pk1"] = 0
    df["pk3"] = 0
    mask = df["k"] <= 0.4
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4", "nk"]]
    return masked.astype(np.float32)


def getwin(winfile):
    # Hardcoded based on Florian's numbers
    kmin = 0.0
    kmax = 0.4
    nkth = 400
    nkout = 40
    step_size = 1

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
    return matrix


def prerecon_sortfunc(item):
    if "UNIT" in item:
        return 0
    else:
        return int(item.split("_")[5])


def postrecon_sortfunc(item):
    if "UNIT" in item:
        return 0
    else:
        return int(item.split("_")[6])


if __name__ == "__main__":

    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/lightcone_mocks_recon/"
    winfile = ds + "/WinMatrix/W_UNIT_NGC_1_1_1_1_1_10_200_2000_averaged_v1.matrix"

    # Power Spectra pre recon
    ds_prerecon = [ds + "preRec/UNIT_Pk/", ds + "preRec/EZ_Pk/"]
    files_prerecon = [d + f for d in ds_prerecon for f in os.listdir(d) if "with_errors" not in f]
    files_prerecon.sort(key=prerecon_sortfunc)
    for file in files_prerecon:
        print(file)
    cov_filename = ds + f"Julian_RecIso/EZ_Cov/cov_julian_reciso_p0p2p4.txt"
    res_prerecon = {f.lower(): getpk(f) for f in files_prerecon}
    ks = next(iter(res_prerecon.items()))[1]["k"].to_numpy()
    cov_input = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    nin = int(len(cov_input) / 3)
    cov_prerecon = np.zeros((5 * len(ks), 5 * len(ks)))
    cov_prerecon[: len(ks), : len(ks)] = np.diag(np.diag(cov_input[: len(ks), : len(ks)]))
    cov_prerecon[: len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[: len(ks), nin : nin + len(ks)]))
    cov_prerecon[: len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov_prerecon[2 * len(ks) : 3 * len(ks), : len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), : len(ks)]))
    cov_prerecon[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(
        np.diag(cov_input[nin : nin + len(ks), nin : nin + len(ks)])
    )
    cov_prerecon[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov_prerecon[4 * len(ks) :, : len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]))
    cov_prerecon[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]))
    cov_prerecon[4 * len(ks) :, 4 * len(ks) :] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]))

    # Power Spectra Julian RecIso
    ds_postrecon = [ds + "Julian_RecIso/UNIT_Pk/", ds + "Julian_RecIso/EZ_Pk/"]
    files_postrecon = [d + f for d in ds_postrecon for f in os.listdir(d) if "with_errors" not in f]
    files_postrecon.sort(key=postrecon_sortfunc)
    for file in files_postrecon:
        print(file)

    cov_filename = ds + f"Julian_RecIso/EZ_Cov/cov_julian_reciso_block_diag_p0p2p4.txt"
    res = {f.lower(): getpk(f) for f in files_postrecon}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov_input = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    nin = int(len(cov_input) / 3)
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = np.diag(np.diag(cov_input[: len(ks), : len(ks)]))
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[: len(ks), nin : nin + len(ks)]))
    cov[: len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), : len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), nin : nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[4 * len(ks) :, : len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]))
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]))
    cov[4 * len(ks) :, 4 * len(ks) :] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    print(np.shape(cov))

    split = {
        "pre-recon data": [v for k, v in res_prerecon.items() if "unit" in k],
        "pre-recon cov": cov_prerecon,
        "post-recon data": [v for k, v in res.items() if "unit" in k],
        "post-recon cov": cov,
        "pre-recon mocks": [v for k, v in res_prerecon.items() if "ezmock" in k],
        "post-recon mocks": [v for k, v in res.items() if "ezmock" in k],
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Lightcone Mocks Recon Julian RecIso",
        "winfit": getwin(winfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../desi_lightcone_mocks_recon_julian_reciso.pkl", "wb") as f:
        pickle.dump(split, f)

    # ==========================================
    # Power Spectra Julian RecSym
    ds_postrecon = [ds + "Julian_RecSym/UNIT_Pk/", ds + "Julian_RecSym/EZ_Pk/"]
    files_postrecon = [d + f for d in ds_postrecon for f in os.listdir(d) if "with_errors" not in f]
    files_postrecon.sort(key=postrecon_sortfunc)
    for file in files_postrecon:
        print(file)

    cov_filename = ds + f"Julian_RecSym/EZ_Cov/cov_julian_recsym_p0p2p4.txt"
    res = {f.lower(): getpk(f) for f in files_postrecon}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov_input = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    nin = int(len(cov_input) / 3)
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = np.diag(np.diag(cov_input[: len(ks), : len(ks)]))
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[: len(ks), nin : nin + len(ks)]))
    cov[: len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), : len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), nin : nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[4 * len(ks) :, : len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]))
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]))
    cov[4 * len(ks) :, 4 * len(ks) :] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    print(np.shape(cov))

    split = {
        "pre-recon data": [v for k, v in res_prerecon.items() if "unit" in k],
        "pre-recon cov": cov_prerecon,
        "post-recon data": [v for k, v in res.items() if "unit" in k],
        "post-recon cov": cov,
        "pre-recon mocks": [v for k, v in res_prerecon.items() if "ezmock" in k],
        "post-recon mocks": [v for k, v in res.items() if "ezmock" in k],
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Lightcone Mocks Recon Julian RecSym",
        "winfit": getwin(winfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../desi_lightcone_mocks_recon_julian_recsym.pkl", "wb") as f:
        pickle.dump(split, f)

    # ==========================================
    # Power Spectra Martin RecIso
    ds_postrecon = [ds + "Martin_RecIso/UNIT_Pk/", ds + "Martin_RecIso/EZ_Pk/"]
    files_postrecon = [d + f for d in ds_postrecon for f in os.listdir(d) if "with_errors" not in f]
    files_postrecon.sort(key=postrecon_sortfunc)
    for file in files_postrecon:
        print(file)

    cov_filename = ds + f"Martin_RecIso/EZ_Cov/cov_martin_reciso_p0p2p4.txt"
    res = {f.lower(): getpk(f) for f in files_postrecon}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov_input = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    nin = int(len(cov_input) / 3)
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = np.diag(np.diag(cov_input[: len(ks), : len(ks)]))
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[: len(ks), nin : nin + len(ks)]))
    cov[: len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), : len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), nin : nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[4 * len(ks) :, : len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]))
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]))
    cov[4 * len(ks) :, 4 * len(ks) :] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    print(np.shape(cov))

    split = {
        "pre-recon data": [v for k, v in res_prerecon.items() if "unit" in k],
        "pre-recon cov": cov_prerecon,
        "post-recon data": [v for k, v in res.items() if "unit" in k and "bugfix" in k],
        "post-recon cov": cov,
        "pre-recon mocks": [v for k, v in res_prerecon.items() if "ezmock" in k],
        "post-recon mocks": [v for k, v in res.items() if "ezmock" in k],
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Lightcone Mocks Recon Martin RecIso",
        "winfit": getwin(winfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../desi_lightcone_mocks_recon_martin_reciso.pkl", "wb") as f:
        pickle.dump(split, f)

    # ==========================================
    # Power Spectra Martin RecSym
    ds_postrecon = [ds + "Martin_RecSym/UNIT_Pk/", ds + "Martin_RecSym/EZ_Pk/"]
    files_postrecon = [d + f for d in ds_postrecon for f in os.listdir(d) if "with_errors" not in f]
    files_postrecon.sort(key=postrecon_sortfunc)
    for file in files_postrecon:
        print(file)

    cov_filename = ds + f"Martin_RecSym/EZ_Cov/cov_martin_recsym_p0p2p4_new.txt"
    res = {f.lower(): getpk(f) for f in files_postrecon}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov_input = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    nin = int(len(cov_input) / 3)
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = np.diag(np.diag(cov_input[: len(ks), : len(ks)]))
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[: len(ks), nin : nin + len(ks)]))
    cov[: len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), : len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[nin : nin + len(ks), nin : nin + len(ks)]))
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = np.diag(np.diag(cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    cov[4 * len(ks) :, : len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]))
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]))
    cov[4 * len(ks) :, 4 * len(ks) :] = np.diag(np.diag(cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]))
    print(np.shape(cov))

    split = {
        "pre-recon data": [v for k, v in res_prerecon.items() if "unit"],
        "pre-recon cov": cov_prerecon,
        "post-recon data": [v for k, v in res.items() if "unit" in k and "bugfix" in k],
        "post-recon cov": cov,
        "pre-recon mocks": [v for k, v in res_prerecon.items() if "ezmock" in k],
        "post-recon mocks": [v for k, v in res.items() if "ezmock" in k],
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Lightcone Mocks Recon Martin RecSym",
        "winfit": getwin(winfile),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(),
    }

    with open(f"../desi_lightcone_mocks_recon_martin_recsym.pkl", "wb") as f:
        pickle.dump(split, f)
