import numpy as np
import matplotlib.pyplot as plt
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.config import setup_logging
from barry.models.model import Correction
from barry.models.bao_correlation_Chen2019 import CorrChen2019
from barry.models.bao_power_Chen2019 import PowerChen2019
from scipy.special import hyp2f1


def D_of_a(a, OmegaM=1):
    return a * hyp2f1(1.0 / 3, 1, 11.0 / 6, -(a**3) / OmegaM * (1 - OmegaM)) / hyp2f1(1.0 / 3, 1, 11.0 / 6, -1 / OmegaM * (1 - OmegaM))


setup_logging()

dataset_pk = PowerSpectrum_DESI_KP4(
    recon="sym",
    fit_poles=[0, 2],
    min_k=0.0,
    max_k=0.3,
    num_mocks=1000,
    reduce_cov_factor=25,
)

# Read in Stephen's model
zelk, zelpk0, zelpk2 = np.loadtxt("../barry/data/desi_kp4/zeldovich_pells.txt")
poly = np.loadtxt("../barry/data/desi_kp4/polynomial_coeffs.txt")

model_pk = PowerChen2019(
    recon=dataset_pk.recon,
    isotropic=dataset_pk.isotropic,
    marg=None,
    fix_params=["om", "sigma_s"],
    poly_poles=dataset_pk.fit_poles,
    correction=Correction.HARTLAP,
)

# Set some default values for the model and compare to Stephen's model, which uses Om=0.30, r_drag=100.
data = dataset_pk.get_data()
data[0]["cosmology"]["om"] = 0.30
data[0]["ks_input"] = zelk
model_pk.set_data(data)

# Load in a pre-existing BAO template
Dz = D_of_a(1.0 / (1.0 + data[0]["cosmology"]["z"]), OmegaM=data[0]["cosmology"]["om"])
pklin = np.loadtxt("../barry/data/desi_kp4/desi_pk.txt")
pknw = np.loadtxt("../barry/data/desi_kp4/desi_pnw.txt")
model_pk.overwrite_template(ks=pklin[:, 0], pklin=Dz**2 * pklin[:, 1], pknw=Dz**2 * pknw[:, 1], r_drag=100.0)

# This function returns the values of k'(k,mu), pk (technically only the unmarginalised terms)
# and model components for analytically marginalised parameters
p = model_pk.get_param_dict(model_pk.get_defaults())
p["alpha"] = 1.0
p["epsilon"] = 0.0
p["b{0}_{1}"] = 2.145
p["beta"] = 0.780 / 2.145
p["sigma_s"] = 0.0
print(model_pk.get_alphas(p["alpha"], p["epsilon"]), p)
print(
    2.0 * model_pk.get_pregen("sigma_dd_nl", p["om"]),
    2.0 * model_pk.get_pregen("sigma_ss_nl", p["om"]),
    2.0 * model_pk.get_pregen("sigma_sd_nl", p["om"]),
)
kprime, pk, marged = model_pk.compute_power_spectrum(data[0]["ks_input"], p, smooth=False, data_name=data[0]["name"])
pk[0] += np.sum([poly[i] * (data[0]["ks_input"] / 0.1) ** (i - 1) for i in range(8)], axis=0)
pk[2] += np.sum([poly[i + 8] * (data[0]["ks_input"] / 0.1) ** (i - 1) for i in range(8)], axis=0)

plt.plot(data[0]["ks_input"], 100.0 * (pk[0] / zelpk0 - 1.0), ls="-", c="r", label="pk0")
plt.plot(data[0]["ks_input"], 100.0 * (pk[2] / zelpk2 - 1.0), ls="-", c="b", label="pk2")
# plt.plot(data[0]["ks_input"], pk[0], ls="-", c="r", label="pk0")
# plt.plot(data[0]["ks_input"], pk[2], ls="-", c="b", label="pk2")
# plt.plot(data[0]["ks_input"], zelpk0, ls="-", c="r", label="pk0")
# plt.plot(data[0]["ks_input"], zelpk2, ls="-", c="b", label="pk2")
plt.ylim(-1.0, 1.0)
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P_{\mathrm{Cullan}}(k)/P_{\mathrm{Stephen}}(k)\,(\%)$")
plt.legend()
plt.tight_layout()
plt.show()

# Now try a fit
model_pk = PowerChen2019(
    recon=dataset_pk.recon,
    isotropic=dataset_pk.isotropic,
    marg="full",
    fix_params=["om", "sigma_s"],
    poly_poles=dataset_pk.fit_poles,
    correction=Correction.HARTLAP,
    n_poly=7,
)

# Set some default values for the model and compare to Stephen's model, which uses Om=0.30, r_drag=100.
data = dataset_pk.get_data()
data[0]["cosmology"]["om"] = 0.30
model_pk.set_data(data)
model_pk.overwrite_template(ks=pklin[:, 0], pklin=Dz**2 * pklin[:, 1], pknw=Dz**2 * pknw[:, 1], r_drag=100.0)
model_pk.sanity_check(dataset_pk)
