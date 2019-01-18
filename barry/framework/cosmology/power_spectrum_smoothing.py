import logging
import math
import numpy as np
from scipy import integrate, interpolate, optimize

import sys
sys.path.append("../../..")


# Smooth power spectrum based on Hinton 2017 polynomial method
def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5):
    # logging.debug("Smoothing spectrum using Hinton 2017 method")
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed

# Smooth power spectrum based on Eisenstein and Hu 1998 fitting formulae for the transfer function 
# with shape of matter power spectrum fit using 5th order polynomial
def smooth_eh1998(ks, pk, om=0.3121, ob=0.0491, h0=0.6751, ns=0.9653, sigma8=0.8150, rs=None):

    # logging.debug("Smoothing spectrum using Eisenstein and Hu 1998 plus 5th order polynomial method")

     # First compute the normalised Eisenstein and Hu smooth power spectrum
    pk_EH98 = ks**ns*__EH98_dewiggled(ks, om, ob, h0, rs)**2
    pk_EH98_spline = interpolate.splrep(ks, pk_EH98)
    pk_EH98_norm = math.sqrt(integrate.quad(__sigma8_integrand,ks[0],ks[-1],args=(ks[0],ks[-1],pk_EH98_spline))[0]/(2.0*math.pi*math.pi))
    pk_EH98 *= (sigma8/pk_EH98_norm)**2

    nll = lambda *args: __EH98_lnlike(*args)
    start = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = optimize.minimize(nll, start, args=(ks, pk_EH98, pk), method="Nelder-Mead", tol=1.0e-6, options={'maxiter': 1000000})

    # Then compute the smooth model
    Apoly = result['x'][1]*ks + result['x'][2] + result['x'][3]/ks + result['x'][4]/ks**2 + result['x'][5]/ks**3 

    return result['x'][0]*pk_EH98+Apoly

# Compute the Eisenstein and Hu dewiggled transfer function
def __EH98_dewiggled(ks, om, ob, h0, rs):

    if (rs == None):
        rs = __EH98_rs(om, ob, h0)

    # Fitting parameters
    a1 = 0.328
    a2 = 431.0
    a3 = 0.380
    a4 = 22.30
    g1 = 0.43
    g2 = 4.0
    c1 = 14.2
    c2 = 731.0
    c3 = 62.5
    l1 = 2.0
    l2 = 1.8
    t1 = 2.0
    theta = 2.725 / 2.7     # Normalised CMB temperature

    q0 = ks*theta*theta
    alpha = 1.0 - a1*math.log(a2*om*h0*h0)*(ob/om) + a3*math.log(a4*om*h0*h0)*(ob/om)**2
    gamma_p1 = (1.0 - alpha)/(1.0 + (g1*ks*rs*h0)**g2)
    gamma = om*h0*(alpha + gamma_p1)
    q = q0/gamma
    c = c1 + c2/(1.0 + c3*q)
    l = np.log(l1*math.exp(1.0) + l2*q)
    t = l/(l + c*q**t1)

    return t

def __EH98_lnlike(params, ks, pkEH, pkdata): 

    pk_B, pk_a1, pk_a2, pk_a3, pk_a4, pk_a5 = params

    Apoly = pk_a1*ks + pk_a2 + pk_a3/ks + pk_a4/ks**2 + pk_a5/ks**3 
    pkfit = pk_B*pkEH+Apoly

    # Compute the chi_squared
    chi_squared = np.sum(((pkdata-pkfit)/pkdata)**2)

    return chi_squared

def __sigma8_integrand(ks, kmin, kmax, pkspline):
    if ((ks < kmin) or (ks > kmax)):
        pk = 0.0
    else:
        pk = interpolate.splev(ks, pkspline, der=0)
    window = 3.0*((math.sin(8.0*ks)/(8.0*ks)**3)-(math.cos(8.0*ks)/(8.0*ks)**2))
    return ks*ks*window*window*pk

# Compute the Eisenstein and Hu 1998 value for the sound horizon
def __EH98_rs(om, ob, h0):

    # Fitting parameters
    b1 = 0.313
    b2 = -0.419
    b3 = 0.607
    b4 = 0.674
    b5 = 0.238
    b6 = 0.223
    a1 = 1291.0
    a2 = 0.251
    a3 = 0.659
    a4 = 0.828
    theta = 2.725 / 2.7     # Normalised CMB temperature

    obh2 = ob*h0*h0
    omh2 = om*h0*h0

    z_eq = 2.5e4*omh2/(theta**4)
    k_eq = 7.46e-2*omh2/(theta**2)

    zd1 = b1*omh2**b2*(1.0 + b3*omh2**b4)
    zd2 = b5*omh2**b6
    z_d = a1*(omh2**a2/(1.0+a3*omh2**a4))*(1.0 + zd1*obh2**zd2)

    R_eq = 3.15e4*obh2/(z_eq*theta**4)
    R_d = 3.15e4*obh2/(z_d*theta**4)

    s = 2.0/(3.0*k_eq)*math.sqrt(6.0/R_eq)*math.log((math.sqrt(1.0+R_d) + math.sqrt(R_d+R_eq))/(1.0 + math.sqrt(R_eq)))

    return s

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")

    om = 0.3121
    h0= 0.6751

    from barry.framework.cosmology.camb_generator import CambGenerator
    camb = CambGenerator(h0=h0)
    ks = camb.ks
    pk_lin = camb.get_data(om=om)

    print(smooth_eh1998(ks, pk_lin))

    import timeit
    n = 500

    def test_hinton():
        smooth_hinton2017(ks, pk_lin)
    def test_eh1998():
        smooth_eh1998(ks, pk_lin)
    #print("Hinton smoothing takes on average, %.2f milliseconds" % (timeit.timeit(test_hinton, number=n) * 1000 / n))
    #print("Eisenstein and Hu smoothing takes on average, %.2f milliseconds" % (timeit.timeit(test_eh1998, number=n) * 1000 / n))

    if True:
        pk_smoothed = smooth_hinton2017(ks, pk_lin)
        pk_smoothed2 = smooth_eh1998(ks, pk_lin)
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(ks, pk_lin, '-', c='k')
        ax1.plot(ks, pk_smoothed, '.', c='r')
        ax1.plot(ks, pk_smoothed2, '+', c='b')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.plot(ks, pk_lin/pk_smoothed, '.', c='r')
        ax2.plot(ks, pk_lin/pk_smoothed2, '+', c='b')
        plt.show()
