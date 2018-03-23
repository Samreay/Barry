import camb
import numpy as np
import matplotlib.pyplot as plt
import timeit

from camb import model, initialpower
print('CAMB version: %s '%camb.__version__)

def fn():

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.set_dark_energy() #re-set defaults
    pars.InitPower.set_params(ns=0.965)
    #Not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.8], kmax=2.0)

    #Linear spectra
    # pars.NonLinear = model.NonLinear_none
    # results = camb.get_results(pars)
    # kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    # s8 = np.array(results.get_sigma8())

    # #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

    # for i, (redshift, line) in enumerate(zip(z,['-','--'])):
    #     plt.loglog(kh, pk[i,:], color='k', ls = line)
    #     plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls = line)
    # plt.xlabel('k/h Mpc');
    # plt.legend(['linear','non-linear'], loc='lower left');
    # plt.title('Matter power at z=%s and z= %s'%tuple(z));

    plt.show()

t = timeit.timeit(fn, number=3) / 3.0

print(t)