import pickle
import pandas as pd
import numpy as np
import os


def getxi(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["s", "xi0", "xi2", "xi4"], header=None)
    mask = df["s"] <= 205.0
    masked = df.loc[mask, ["s", "xi0", "xi2", "xi4"]]
    print(masked)
    return masked.astype(np.float32)


def getpk_pre(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["k", "pk0", "pk2", "pk4", "nk"], header=None)
    df["pk1"] = 0
    df["pk3"] = 0
    mask = df["k"] <= 0.5
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4"]]
    return masked.astype(np.float32)


def getpk_post(loc):
    df = pd.read_csv(loc, comment="#", skiprows=0, delim_whitespace=True, names=["k", "pk0", "pk2", "pk4"], header=None)
    df["pk1"] = 0
    df["pk3"] = 0
    mask = df["k"] <= 0.5
    masked = df.loc[mask, ["k", "pk0", "pk1", "pk2", "pk3", "pk4"]]
    print(masked)
    return masked.astype(np.float32)


def getwin(ks):
    res = {"w_ks_input": ks.copy(), "w_k0_scale": np.zeros(ks.size), "w_transform": np.eye(5 * ks.size), "w_ks_output": ks.copy()}
    return {1: res}  # Step size is one


def getcomp(ks):
    matrix = np.zeros((5 * ks.size, 3 * ks.size))
    matrix[: ks.size, : ks.size] = np.diag(np.ones(ks.size))
    matrix[2 * ks.size : 3 * ks.size, ks.size : 2 * ks.size] = np.diag(np.ones(ks.size))
    matrix[4 * ks.size :, 2 * ks.size :] = np.diag(np.ones(ks.size))
    return matrix


if __name__ == "__main__":

    # Power Spectra Pre Recon Standard
    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO_stage_2/PreRecon/"

    files = [ds + "UNIT_3Gpc_Nmesh512_pkl.txt"]
    print(files)

    cov_filename = ds + f"/cov_matrix_pk-EZmocks-3Gpc-nonfix_rsd_centerbin.txt"
    res = {f.lower(): getpk_pre(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    cov_flat = cov.astype(np.float64)[:, 2]
    nin = int(np.sqrt(len(cov)) / 3)
    cov_input = cov_flat.reshape((3 * nin, 3 * nin))
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = cov_input[: len(ks), : len(ks)]
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[: len(ks), nin : nin + len(ks)]
    cov[: len(ks), 4 * len(ks) :] = cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = cov_input[nin : nin + len(ks), : len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[nin : nin + len(ks), nin : nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[4 * len(ks) :, : len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]
    cov[4 * len(ks) :, 4 * len(ks) :] = cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]

    print(np.shape(cov), np.shape(ks))

    split = {
        "pre-recon data": None,
        "pre-recon cov": cov,
        "post-recon data": None,
        "post-recon cov": None,
        "pre-recon mocks": [v for k, v in res.items()],
        "post-recon mocks": None,
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Mock Challenge Stage 2 Pk Std",
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../desi_mock_challenge_post_stage_2_pk_pre_nonfix.pkl", "wb") as f:
        pickle.dump(split, f)

    # Power Spectra Pre Recon Fixed Amplitude
    files = [ds + "UNIT_3Gpc_Nmesh512_pkl.txt"]
    print(files)

    cov_filename = ds + f"/cov_matrix_pk-EZmocks-3Gpc_rsd_centerbin.txt"
    res = {f.lower(): getpk_pre(f) for f in files}
    ks = next(iter(res.items()))[1]["k"].to_numpy()
    cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
    cov_flat = cov.astype(np.float64)[:, 2]
    nin = int(np.sqrt(len(cov)) / 3)
    cov_input = cov_flat.reshape((3 * nin, 3 * nin))
    cov = np.zeros((5 * len(ks), 5 * len(ks)))
    cov[: len(ks), : len(ks)] = cov_input[: len(ks), : len(ks)]
    cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[: len(ks), nin : nin + len(ks)]
    cov[: len(ks), 4 * len(ks) :] = cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), : len(ks)] = cov_input[nin : nin + len(ks), : len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[nin : nin + len(ks), nin : nin + len(ks)]
    cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]
    cov[4 * len(ks) :, : len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]
    cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]
    cov[4 * len(ks) :, 4 * len(ks) :] = cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]

    print(np.shape(cov), np.shape(ks))

    split = {
        "pre-recon data": None,
        "pre-recon cov": cov,
        "post-recon data": None,
        "post-recon cov": None,
        "pre-recon mocks": [v for k, v in res.items()],
        "post-recon mocks": None,
        "cosmology": {
            "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
            "h0": 0.6774,
            "z": 0.9873,
            "ob": 0.02230 / 0.6774 ** 2,
            "ns": 0.9667,
            "mnu": 0.00064 * 93.14,
            "reconsmoothscale": 15,
        },
        "name": f"DESI Mock Challenge Stage 2 Pk Fix",
        "winfit": getwin(ks),
        "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
        "m_mat": getcomp(ks),
    }

    with open(f"../desi_mock_challenge_post_stage_2_pk_pre.pkl", "wb") as f:
        pickle.dump(split, f)

    # ========================================
    # Power Spectra Post Recon Iso
    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO_stage_2/RecIso/"

    files = [
        [
            ds + "UNIELG-b0s05rsd0g1536postmultipoles.txt",
        ],
        [
            ds + "UNIELG-b0s10rsd0g1536postmultipoles.txt",
        ],
        [
            ds + "UNIELG-b0s15rsd0g1536postmultipoles.txt",
        ],
        [
            ds + "UNIELG-b0s20rsd0g1536postmultipoles.txt",
        ],
    ]

    covfiles = [f"/cov_matrix_pk-EZmocks-3Gpc_rsd_centerbin_RecIso_post.txt", f"/cov_matrix_pk-EZmocks-3Gpc-RecIso-nonfix_rsd_post.txt"]
    for i, smooth in enumerate([5, 10, 15, 20]):

        for j, fix in enumerate(["", "_nonfix"]):

            print(files[i])
            print(covfiles[j])

            cov_filename = ds + covfiles[j]
            res = {f.lower(): getpk_post(f) for f in files[i]}
            ks = next(iter(res.items()))[1]["k"].to_numpy()
            cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
            cov_flat = cov.astype(np.float64)[:, 2]
            nin = int(np.sqrt(len(cov)) / 3)
            cov_input = cov_flat.reshape((3 * nin, 3 * nin))
            cov = np.zeros((5 * len(ks), 5 * len(ks)))
            cov[: len(ks), : len(ks)] = cov_input[: len(ks), : len(ks)]
            cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[: len(ks), nin : nin + len(ks)]
            cov[: len(ks), 4 * len(ks) :] = cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]
            cov[2 * len(ks) : 3 * len(ks), : len(ks)] = cov_input[nin : nin + len(ks), : len(ks)]
            cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[nin : nin + len(ks), nin : nin + len(ks)]
            cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]
            cov[4 * len(ks) :, : len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]
            cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]
            cov[4 * len(ks) :, 4 * len(ks) :] = cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]

            print(np.shape(cov), np.shape(ks))

            split = {
                "pre-recon data": None,
                "pre-recon cov": None,
                "post-recon data": None,
                "post-recon cov": cov,
                "pre-recon mocks": None,
                "post-recon mocks": [v for k, v in res.items()],
                "cosmology": {
                    "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
                    "h0": 0.6774,
                    "z": 0.9873,
                    "ob": 0.02230 / 0.6774 ** 2,
                    "ns": 0.9667,
                    "mnu": 0.00064 * 93.14,
                    "reconsmoothscale": smooth,
                },
                "name": f"DESI Mock Challenge Stage 2 Pk RecIso " + str(smooth) + fix,
                "winfit": getwin(ks),
                "winpk": None,  # We can set this to None; Barry will set it to zeroes given the length of the data vector.
                "m_mat": getcomp(ks),
            }

            with open(f"../desi_mock_challenge_post_stage_2_pk_iso_" + str(smooth) + fix + ".pkl", "wb") as f:
                pickle.dump(split, f)

    # ==========================================
    # Power Spectra Post Recon Ani
    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO_stage_2/RecAni/"

    files = [
        [ds + "UNIELG-b0s05rsd0g1536postmultipoles.txt"],
        [ds + "UNIELG-b0s10rsd0g1536postmultipoles.txt"],
        [ds + "UNIELG-b0s15rsd0g1536postmultipoles.txt"],
        [ds + "UNIELG-b0s20rsd0g1536postmultipoles.txt"],
    ]

    covfiles = [f"/cov_matrix_pk-EZmocks-3Gpc_rsd_centerbin_post.txt", f"/cov_matrix_pk-EZmocks-3Gpc-RecAni-nonfix_rsd_post.txt"]

    for i, smooth in enumerate([5, 10, 15, 20]):

        for j, fix in enumerate(["", "_nonfix"]):

            print(files[i])
            print(covfiles[j])

            cov_filename = ds + covfiles[j]
            res = {f.lower(): getpk_post(f) for f in files[i]}
            ks = next(iter(res.items()))[1]["k"].to_numpy()
            cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
            cov_flat = cov.astype(np.float64)[:, 2]
            nin = int(np.sqrt(len(cov)) / 3)
            cov_input = cov_flat.reshape((3 * nin, 3 * nin))
            cov = np.zeros((5 * len(ks), 5 * len(ks)))
            cov[: len(ks), : len(ks)] = cov_input[: len(ks), : len(ks)]
            cov[: len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[: len(ks), nin : nin + len(ks)]
            cov[: len(ks), 4 * len(ks) :] = cov_input[: len(ks), 2 * nin : 2 * nin + len(ks)]
            cov[2 * len(ks) : 3 * len(ks), : len(ks)] = cov_input[nin : nin + len(ks), : len(ks)]
            cov[2 * len(ks) : 3 * len(ks), 2 * len(ks) : 3 * len(ks)] = cov_input[nin : nin + len(ks), nin : nin + len(ks)]
            cov[2 * len(ks) : 3 * len(ks), 4 * len(ks) :] = cov_input[nin : nin + len(ks), 2 * nin : 2 * nin + len(ks)]
            cov[4 * len(ks) :, : len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), : len(ks)]
            cov[4 * len(ks) :, 2 * len(ks) : 3 * len(ks)] = cov_input[2 * nin : 2 * nin + len(ks), nin : nin + len(ks)]
            cov[4 * len(ks) :, 4 * len(ks) :] = cov_input[2 * nin : 2 * nin + len(ks), 2 * nin : 2 * nin + len(ks)]

            print(np.shape(cov), np.shape(ks))

            split = {
                "pre-recon data": None,
                "pre-recon cov": None,
                "post-recon data": None,
                "post-recon cov": cov,
                "pre-recon mocks": None,
                "post-recon mocks": [v for k, v in res.items()],
                "cosmology": {
                    "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
                    "h0": 0.6774,
                    "z": 0.9873,
                    "ob": 0.02230 / 0.6774 ** 2,
                    "ns": 0.9667,
                    "mnu": 0.00064 * 93.14,
                    "reconsmoothscale": smooth,
                },
                "name": f"DESI Mock Challenge Stage 2 Pk RecAni " + str(smooth) + fix,
                "winfit": getwin(ks),
                "winpk": None,
                # We can set this to None; Barry will set it to zeroes given the length of the data vector.
                "m_mat": getcomp(ks),
            }

            with open(f"../desi_mock_challenge_post_stage_2_pk_ani_" + str(smooth) + fix + ".pkl", "wb") as f:
                pickle.dump(split, f)

    # ===========================================
    # Correlation Function Post Recon Iso
    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO_stage_2/RecIso/"

    files = [
        [ds + "UNIELG-b0s05rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s10rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s15rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s20rsd0g1536postmultipoles_xi_gs_han4.txt"],
    ]

    covfiles = [
        f"/cov_matrix_xi-EZmocks-3Gpc_rsd_centerbin_post_mariana.txt",
        f"/cov_matrix_xi-EZmocks-3Gpc-RecIso-nonfix_rsd_post_bin-center.txt",
    ]

    for i, smooth in enumerate([5, 10, 15, 20]):

        for j, fix in enumerate(["", "_nonfix"]):

            print(files[i])
            print(covfiles[j])

            cov_filename = ds + covfiles[j]
            res = {f.lower(): getxi(f) for f in files[i]}
            start = 0
            nss = len(next(iter(res.items()))[1]["s"].to_numpy())
            cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
            cov_flat = cov.astype(np.float32)[:, 2]
            nin = int(np.sqrt(len(cov)) / 3)
            cov_input = cov_flat.reshape((3 * nin, 3 * nin))
            cov = np.zeros((3 * nss, 3 * nss))
            cov[:nss, :nss] = cov_input[start : start + nss, start : start + nss]
            cov[:nss, nss : 2 * nss] = cov_input[start : start + nss, start + nin : start + nin + nss]
            cov[:nss, 2 * nss :] = cov_input[start : start + nss, start + 2 * nin : start + 2 * nin + nss]
            cov[nss : 2 * nss, :nss] = cov_input[start + nin : start + nin + nss, start : start + nss]
            cov[nss : 2 * nss, nss : 2 * nss] = cov_input[start + nin : start + nin + nss, start + nin : start + nin + nss]
            cov[nss : 2 * nss, 2 * nss :] = cov_input[start + nin : start + nin + nss, start + 2 * nin : start + 2 * nin + nss]
            cov[2 * nss :, :nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start : start + nss]
            cov[2 * nss :, nss : 2 * nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + nin : start + nin + nss]
            cov[2 * nss :, 2 * nss :] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + 2 * nin : start + 2 * nin + nss]

            print(np.shape(cov), nss)

            split = {
                "pre-recon data": None,
                "pre-recon cov": None,
                "post-recon data": None,
                "post-recon cov": cov,
                "pre-recon mocks": None,
                "post-recon mocks": [v for k, v in res.items()],
                "cosmology": {
                    "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
                    "h0": 0.6774,
                    "z": 0.9873,
                    "ob": 0.02230 / 0.6774 ** 2,
                    "ns": 0.9667,
                    "mnu": 0.00064 * 93.14,
                    "reconsmoothscale": smooth,
                },
                "name": f"DESI Mock Challenge Stage 2 Xi RecIso" + str(smooth) + fix,
            }

            with open(f"../desi_mock_challenge_post_stage_2_xi_iso_" + str(smooth) + fix + ".pkl", "wb") as f:
                pickle.dump(split, f)

    # ===========================================
    # Correlation Function Post Recon Ani
    ds = f"/Volumes/Work/UQ/DESI/MockChallenge/Post_recon_BAO_stage_2/RecAni/"

    files = [
        [ds + "UNIELG-b0s05rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s10rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s15rsd0g1536postmultipoles_xi_gs_han4.txt"],
        [ds + "UNIELG-b0s20rsd0g1536postmultipoles_xi_gs_han4.txt"],
    ]

    covfiles = [
        f"/cov_matrix_xi-EZmocks-3Gpc_rsd_centerbin_post_mariana.txt",
        f"/cov_matrix_xi-EZmocks-3Gpc-RecAni-nonfix_rsd_post_bin-center.txt",
    ]

    for i, smooth in enumerate([5, 10, 15, 20]):

        for j, fix in enumerate(["", "_nonfix"]):
            print(files[i])
            print(covfiles[j])

            cov_filename = ds + covfiles[j]
            res = {f.lower(): getxi(f) for f in files[i]}
            start = 0
            nss = len(next(iter(res.items()))[1]["s"].to_numpy())
            cov = pd.read_csv(cov_filename, delim_whitespace=True, header=None).to_numpy()
            cov_flat = cov.astype(np.float32)[:, 2]
            nin = int(np.sqrt(len(cov)) / 3)
            cov_input = cov_flat.reshape((3 * nin, 3 * nin))
            cov = np.zeros((3 * nss, 3 * nss))
            cov[:nss, :nss] = cov_input[start : start + nss, start : start + nss]
            cov[:nss, nss : 2 * nss] = cov_input[start : start + nss, start + nin : start + nin + nss]
            cov[:nss, 2 * nss :] = cov_input[start : start + nss, start + 2 * nin : start + 2 * nin + nss]
            cov[nss : 2 * nss, :nss] = cov_input[start + nin : start + nin + nss, start : start + nss]
            cov[nss : 2 * nss, nss : 2 * nss] = cov_input[start + nin : start + nin + nss, start + nin : start + nin + nss]
            cov[nss : 2 * nss, 2 * nss :] = cov_input[start + nin : start + nin + nss, start + 2 * nin : start + 2 * nin + nss]
            cov[2 * nss :, :nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start : start + nss]
            cov[2 * nss :, nss : 2 * nss] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + nin : start + nin + nss]
            cov[2 * nss :, 2 * nss :] = cov_input[start + 2 * nin : start + 2 * nin + nss, start + 2 * nin : start + 2 * nin + nss]

            print(np.shape(cov), nss)

            split = {
                "pre-recon data": None,
                "pre-recon cov": None,
                "post-recon data": None,
                "post-recon cov": cov,
                "pre-recon mocks": None,
                "post-recon mocks": [v for k, v in res.items()],
                "cosmology": {
                    "om": (0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2,
                    "h0": 0.6774,
                    "z": 0.9873,
                    "ob": 0.02230 / 0.6774 ** 2,
                    "ns": 0.9667,
                    "mnu": 0.00064 * 93.14,
                    "reconsmoothscale": smooth,
                },
                "name": f"DESI Mock Challenge Stage 2 Xi RecAni" + str(smooth) + fix,
            }

            with open(f"../desi_mock_challenge_post_stage_2_xi_ani_" + str(smooth) + fix + ".pkl", "wb") as f:
                pickle.dump(split, f)
