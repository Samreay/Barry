import pickle
import pandas as pd
import numpy as np
import os


def getdf(loc):
    df = pd.read_csv(loc, comment="#", delim_whitespace=True, names=["s", "xi0"])
    output = df.to_numpy().astype(np.float32)
    return output


ds = ["pre_recon/", "post_recon/"]
files = [sorted([d + f for f in os.listdir(d) if f.endswith("dat")]) for d in ds]
print([f.split("_")[-1][:-4] for f in files[1]])

split = {
    "pre-recon": [getdf(f) for f in files[0]],
    "post-recon": [getdf(f) for f in files[1]],
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "reconsmoothscale": 15},
    "name": "SDSS DR12 Z0.61 Corr NGC",
}
with open("../sdss_dr12_z061_corr_ngc.pkl", "wb") as f:
    pickle.dump(split, f)
