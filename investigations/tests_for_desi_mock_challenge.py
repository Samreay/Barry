import logging
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESIMockChallenge0_Z01
from barry.models import PowerBeutler2017
from barry.models.model import Correction
import pandas as pd
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # First get the bestfit using our fiducial setup
    d = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data", min_k=0.02, max_k=0.30)
    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model.set_default("sigma_nl_par", 10.9)
    model.set_default("sigma_nl_perp", 5.98)
    model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp"])
    model.set_data(d.get_data())
    p, minv = model.optimize(niter=2000, maxiter=20000)
    print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")
    print(model.get_alphas(p["alpha"], p["epsilon"]))
    model.plot(p)

    # Now match Hee-Jong's k-range
    d = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data", min_k=0.001, max_k=0.30)
    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model.set_default("sigma_nl_par", 10.9)
    model.set_default("sigma_nl_perp", 5.98)
    model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp"])
    model.set_data(d.get_data())
    p, minv = model.optimize(niter=2000, maxiter=20000)
    print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")
    print(model.get_alphas(p["alpha"], p["epsilon"]))
    model.plot(p)

    # Now use Hee-Jong's values for the damping parameters
    d = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data", min_k=0.001, max_k=0.30)
    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model.set_default("sigma_nl_par", 6.2)
    model.set_default("sigma_nl_perp", 2.9)
    model.set_default("sigma_s", 0.0)
    model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"])
    model.set_data(d.get_data())
    p, minv = model.optimize(niter=2000, maxiter=20000)
    print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")
    print(model.get_alphas(p["alpha"], p["epsilon"]))
    model.plot(p)

    # Now try Hee-Jong's power spectra
    pklin = np.array(pd.read_csv("/Volumes/Work/UQ/DESI/MockChallenge/mylinearmatterpkL900.dat", delim_whitespace=True, header=None))
    pksmooth = np.array(pd.read_csv("/Volumes/Work/UQ/DESI/MockChallenge/Psh_mylinearmatterpkL900.dat", delim_whitespace=True, header=None, skiprows=2))

    d = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data", min_k=0.001, max_k=0.30)
    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model.set_default("sigma_nl_par", 6.2)
    model.set_default("sigma_nl_perp", 2.9)
    model.set_default("sigma_s", 0.0)
    model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp", "sigma_s"])
    model.set_data(d.get_data())
    model.camb.ks = pklin[:, 0]
    model.pkratio = pklin[:, 1] / pksmooth[:, 1] - 1.0
    model.pksmooth = pksmooth[:, 1]
    p, minv = model.optimize(niter=2000, maxiter=20000)
    print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")
    print(model.get_alphas(p["alpha"], p["epsilon"]))
    model.plot(p)

    # What about Hee-Jong's template, but our k-range and damping?
    pklin = np.array(pd.read_csv("/Volumes/Work/UQ/DESI/MockChallenge/mylinearmatterpkL900.dat", delim_whitespace=True, header=None))
    pksmooth = np.array(pd.read_csv("/Volumes/Work/UQ/DESI/MockChallenge/Psh_mylinearmatterpkL900.dat", delim_whitespace=True, header=None, skiprows=2))

    d = PowerSpectrum_DESIMockChallenge0_Z01(recon=False, isotropic=False, realisation="data", min_k=0.02, max_k=0.30)
    model = PowerBeutler2017(recon=False, isotropic=False, correction=Correction.NONE)
    model.set_default("sigma_nl_par", 10.9)
    model.set_default("sigma_nl_perp", 5.98)
    model.set_fix_params(["om", "sigma_nl_par", "sigma_nl_perp"])
    model.set_data(d.get_data())
    model.camb.ks = pklin[:, 0]
    model.pkratio = pklin[:, 1] / pksmooth[:, 1] - 1.0
    model.pksmooth = pksmooth[:, 1]
    p, minv = model.optimize(niter=2000, maxiter=20000)
    print(f"Model optimisation with value {minv:0.3f} has parameters are {dict(p)}")
    print(model.get_alphas(p["alpha"], p["epsilon"]))
    model.plot(p)
