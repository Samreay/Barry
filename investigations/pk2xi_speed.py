from barry.cosmology import PowerToCorrelationGauss, PowerToCorrelationFT, getCambGenerator
import numpy as np
from timeit import timeit

camb = getCambGenerator()
ks = camb.ks
pk = camb.get_data()[1]
ss = np.linspace(20, 200, 50)
gauss = PowerToCorrelationGauss(ks)
fft = PowerToCorrelationFT()
n = 500


def time_gauss():
    gauss(ks, pk, ss)


def time_ft():
    fft(ks, pk, ss)


print(f"Gaussian method takes {timeit(time_gauss, number=n) * 1000 / n  : 0.2f} milliseconds")
print(f"FFT method takes {timeit(time_ft, number=n) * 1000 / n  : 0.2f} milliseconds")
