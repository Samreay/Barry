import logging
import numpy as np


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
