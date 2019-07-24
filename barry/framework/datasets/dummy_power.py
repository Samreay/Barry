import os
import logging
import inspect
import pickle
import numpy as np
from scipy.interpolate import splev, splrep

from barry.framework.cosmology.camb_generator import CambGenerator
from barry.framework.datasets import MockSDSSdr12PowerSpectrum


class DummyPowerSpectrum(MockSDSSdr12PowerSpectrum):
    def __init__(self, min_k=0.02, max_k=0.30, step_size=5, uncert=0.01,
                 reduce_cov_factor=1, name="DummyPowerSpectrum", postprocess=None,
                 apply_hartlap_correction=False, fake_diag=False, dummy_window=False):
        super().__init__(average=True, min_k=min_k, max_k=max_k, step_size=step_size,
                         recon=True, reduce_cov_factor=reduce_cov_factor, name=name, postprocess=postprocess,
                         apply_hartlap_correction=apply_hartlap_correction, fake_diag=fake_diag)

        # Set data to camb generated power spectrum
        c = CambGenerator()
        r_s, pk_lin = c.get_data()
        ks = c.ks

        # Apply window function identically as it was applied to data
        # Note this isn't perfect because any error in the window function will propagate
        if dummy_window:
            pk_final = splev(self.w_ks_output, splrep(ks, pk_lin))[self.w_mask]
            self.w_ks_input = self.w_ks_output
            self.w_transform = np.diag(np.ones(self.w_ks_output.size))
            self.w_k0_scale = 0
            self.w_pk = 0
        else:
            pk_sampled = splev(self.w_ks_input, splrep(ks, pk_lin))
            p0 = np.sum(self.w_k0_scale * pk_sampled)
            integral_constraint = self.w_pk * p0
            pk_convolved = np.atleast_2d(pk_sampled) @ self.w_transform
            pk_normalised = (pk_convolved - integral_constraint).flatten()
            pk_final = pk_normalised[self.w_mask]

        # Set covariance to something nice and simple to sample from
        # 1% diagonal uncertainty seems pretty good.
        cov = np.diag((uncert * pk_final) ** 2)
        self.data = pk_final
        self.set_cov(cov)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)18s]   %(message)s")

    dataset = DummyPowerSpectrum()
    data = dataset.get_data()
    print(data["ks"])
    print(data["pk"])

    import matplotlib.pyplot as plt
    import seaborn as sb
    import numpy as np
    plt.errorbar(data["ks"], data["ks"]*data["pk"], yerr=data["ks"]*np.sqrt(np.diag(data["cov"])), fmt="o", c='k')
    plt.show()

    sb.heatmap(data["cov"])
    plt.show()
