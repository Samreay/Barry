import numpy as np
import matplotlib.pyplot as plt
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.config import setup_logging
from barry.models.model import Correction
from barry.models.bao_correlation_Chen2019 import CorrChen2019

setup_logging()

dataset = CorrelationFunction_DESI_KP4(
    recon="sym",
    fit_poles=[0, 2],
    min_dist=52.0,
    max_dist=150.0,
    num_mocks=1000,
    reduce_cov_factor=25,
)
data = dataset.get_data()[0]
plt.plot(data["dist"], data["dist"] ** 2 * data["xi0"], c="r", ls="-", label="xi0")
plt.plot(data["dist"], data["dist"] ** 2 * data["xi2"], c="b", ls="-", label="xi2")

# Overwrite the data vector with Stephen's Zeldovich theory curve
indata = np.loadtxt("../barry/data/desi_kp4/lrg_xiells_zeldovich_damping.txt")
index = np.where((indata[:, 0] >= 52.0) & (indata[:, 0] <= 150.0))
dataset.data[:, :-1] = indata[index, 1:]
data = dataset.get_data()[0]

plt.plot(data["dist"], data["dist"] ** 2 * data["xi0"], c="r", ls="--", label="xi0")
plt.plot(data["dist"], data["dist"] ** 2 * data["xi2"], c="b", ls="--", label="xi2")
plt.show()

model = CorrChen2019(
    recon=dataset.recon,
    isotropic=dataset.isotropic,
    marg="full",
    fix_params=["om", "sigma_s"],
    poly_poles=dataset.fit_poles,
    correction=Correction.HARTLAP,
)
model.set_default("sigma_s", 0.0)

# Load in the template and overwrite the standard model one
desi_pk = np.loadtxt("../barry/data/desi_kp4/desi_pk.txt")
desi_pnw = np.loadtxt("../barry/data/desi_kp4/desi_pnw.txt")
plt.plot(desi_pk[:, 0], desi_pk[:, 0] * desi_pk[:, 1], c="r", ls="-", label="pk")
plt.plot(desi_pnw[:, 0], desi_pnw[:, 0] * desi_pnw[:, 1], c="b", ls="-", label="pnw")
plt.xlim(0.0, 0.4)
plt.show()

# model.kvals, model.pksmooth = desi_pnw.T
# model.pkratio = desi_pk[:, 1] / desi_pnw[:, 1] - 1.0

# Run a fit
model.sanity_check(dataset)
