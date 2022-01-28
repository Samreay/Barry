import pickle
import pandas as pd
import numpy as np
import os


def getdata(xi0file, xi2file):
    dfxi0 = pd.read_csv(xi0file, comment="#", delim_whitespace=True, names=["s", "xi0", "xi0err"]).drop("xi0err", axis=1)
    dfxi2 = pd.read_csv(xi2file, comment="#", delim_whitespace=True, names=["s", "xi2", "xi2err"]).drop("xi2err", axis=1)
    df = pd.merge(dfxi0, dfxi2, on="s")
    df["xi4"] = np.zeros(len(df["s"]))
    # output = df.to_numpy().astype(np.float32)
    # print(np.shape(output))
    return df


def getcov(covfile):
    df = pd.read_csv(covfile, comment="#", delim_whitespace=True, header=None)
    nss = int(len(df) / 2)
    output = np.zeros((3 * nss, 3 * nss))
    output[: 2 * nss, : 2 * nss] = df.to_numpy().astype(np.float32)
    print(nss, np.shape(output))
    return output


# Redshift Bin 1 (z_eff = 0.38)
xi0file = "Ross_2016_COMBINEDDR12_zbin1_correlation_function_monopole_post_recon_bincent0.dat"
xi2file = "Ross_2016_COMBINEDDR12_zbin1_correlation_function_quadrupole_post_recon_bincent0.dat"
covfile = "Ross_2016_COMBINEDDR12_zbin1_covariance_monoquad_post_recon_bincent0.dat"

split = {
    "pre-recon data": None,
    "post-recon data": [getdata(xi0file, xi2file)],
    "pre-recon mocks": None,
    "post-recon mocks": None,
    "pre-recon cov": None,
    "post-recon cov": getcov(covfile),
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "mnu": 0.00, "reconsmoothscale": 15},
    "name": "Ross 2016 Combined z038 corr",
}
with open("../ross_2016_dr12_combined_corr_zbin0p38.pkl", "wb") as f:
    pickle.dump(split, f)

# Redshift Bin 2 (z_eff = 0.51)
xi0file = "Ross_2016_COMBINEDDR12_zbin2_correlation_function_monopole_post_recon_bincent0.dat"
xi2file = "Ross_2016_COMBINEDDR12_zbin2_correlation_function_quadrupole_post_recon_bincent0.dat"
covfile = "Ross_2016_COMBINEDDR12_zbin2_covariance_monoquad_post_recon_bincent0.dat"

split = {
    "pre-recon data": None,
    "post-recon data": [getdata(xi0file, xi2file)],
    "pre-recon mocks": None,
    "post-recon mocks": None,
    "pre-recon cov": None,
    "post-recon cov": getcov(covfile),
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "mnu": 0.00, "reconsmoothscale": 15},
    "name": "Ross 2016 Combined z051 corr",
}

with open("../ross_2016_dr12_combined_corr_zbin0p51.pkl", "wb") as f:
    pickle.dump(split, f)

# Redshift Bin 3 (z_eff = 0.61)
xi0file = "Ross_2016_COMBINEDDR12_zbin3_correlation_function_monopole_post_recon_bincent0.dat"
xi2file = "Ross_2016_COMBINEDDR12_zbin3_correlation_function_quadrupole_post_recon_bincent0.dat"
covfile = "Ross_2016_COMBINEDDR12_zbin3_covariance_monoquad_post_recon_bincent0.dat"

split = {
    "pre-recon data": None,
    "post-recon data": [getdata(xi0file, xi2file)],
    "pre-recon mocks": None,
    "post-recon mocks": None,
    "pre-recon cov": None,
    "post-recon cov": getcov(covfile),
    "cosmology": {"om": 0.31, "h0": 0.676, "z": 0.61, "ob": 0.04814, "ns": 0.97, "mnu": 0.00, "reconsmoothscale": 15},
    "name": "Ross 2016 Combined z061 corr",
}
with open("../ross_2016_dr12_combined_corr_zbin0p61.pkl", "wb") as f:
    pickle.dump(split, f)
