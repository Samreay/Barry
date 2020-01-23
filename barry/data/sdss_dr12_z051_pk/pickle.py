import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(loc, comment="#", skiprows=34, delim_whitespace=True, names=["k", "kmean", "pk", "pk2", "pk4", "sigma", "nk"])
    mask = df["nk"] > 0
    masked = df.loc[mask, ["k", "pk", "nk"]]
    return masked.astype(np.float32)


def get_winfits(directory):
    files = sorted([directory + f for f in os.listdir(directory) if "winfit" in f and gc in f])
    print(files)
    winfits = {}
    for winfit_file in files:
        step_size = 1  # int(winfit_file.split(".")[-2].split("_")[-1])
        print(f"Loading winfit from {winfit_file}")
        matrix = np.genfromtxt(winfit_file, skip_header=4)
        res = {}
        winfits[step_size] = res
        res["w_ks_input"] = matrix[:, 0]
        res["w_k0_scale"] = matrix[:, 1]
        res["w_transform"] = matrix[:, 2:] / (np.sum(matrix[:, 2:], axis=0))
        # God I am sorry for doing this manually but the file format is... tricky
        with open(winfit_file, "r") as f:
            res["w_ks_output"] = np.array([float(x) for x in f.readlines()[2].split()[1:]])
    return winfits


def get_winpk(directory):
    files = sorted([directory + f for f in os.listdir(directory) if "lwin" in f and gc in f])
    assert len(files) == 1, str(files)
    print(files[0])
    data = np.genfromtxt(files[0])
    return data


for gc in ["NGC", "SGC"]:
    ds = [f"post_recon {gc}/", f"pre_recon {gc}/"]
    files = [d + f for d in ds for f in os.listdir(d)]
    files.sort(key=lambda s: int(s.split("_z2_")[1].split("_")[0]))
    print(files[:4])

    res = {f.lower(): getdf(f) for f in files}
    split = {
        "pre-recon": [v for k, v in res.items() if "pre_recon" in k],
        "post-recon": [v for k, v in res.items() if "post_recon" in k],
        "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.51, "ob": 0.04814, "ns": 0.97, "reconsmoothscale": 15},
        "name": f"SDSS DR12 Z0.51 Pk {gc}",
        "winfit": get_winfits("./"),
        "winpk": get_winpk("./"),
    }

    with open(f"../sdss_dr12_z051_pk_{gc.lower()}.pkl", "wb") as f:
        pickle.dump(split, f)
