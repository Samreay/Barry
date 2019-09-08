from barry.cosmology import PowerToCorrelationGauss, PowerToCorrelationFT, getCambGenerator
import matplotlib.pyplot as plt
import numpy as np

camb = getCambGenerator()
ks = camb.ks
pk = camb.get_data()[1]
truth = PowerToCorrelationGauss(ks, interpolateDetail=20, a=0.1)

ss = np.linspace(20, 200, 200)
xi_truth = truth(camb.ks, pk, ss)
gauss = PowerToCorrelationGauss(ks)
fft = PowerToCorrelationFT()

xi_bad = PowerToCorrelationGauss(ks, interpolateDetail=1, a=0)(ks, pk, ss)
xi_gauss = gauss(ks, pk, ss)
xi_ft = fft(ks, pk, ss)

fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
for xi, l in [(xi_bad, "Bad"), (xi_gauss, "Gauss"), (xi_ft, "FT"), (xi_truth, "Truth")]:
    axes[0].plot(ss, ss ** 2 * xi, label=l, ls="--" if l == "Truth" else "-", lw=1)
    axes[1].plot(ss, ss ** 2 * (xi - xi_truth), label=l, ls="--" if l == "Truth" else "-", lw=1)

axes[0].legend(), axes[1].legend()
plt.show()
