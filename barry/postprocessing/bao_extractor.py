import logging

import numpy as np
from barry.postprocessing.postprocessor import PkPostProcess


class PureBAOExtractor(PkPostProcess):
    """The pure BAO extractor detailed in Noda 2017 (1705.01475), Nishimishi 2018 (1708.00375), Noda 2019 (1901.06854)

    See https://ui.adsabs.harvard.edu/abs/2017JCAP...08..007N
    See https://ui.adsabs.harvard.edu/abs/2018JCAP...01..035N
    See https://ui.adsabs.harvard.edu/abs/2019arXiv190106854N

    Parameters
    ----------
    r_s : float
        The sound horizon distance. In units of Mpc/h
    plot : bool, optional
        Whether to output debugging plots
    delta : float, optional
        The window (in units of `r_s` to smooth)
    """

    def __init__(self, r_s, plot=False, delta=0.6):
        super().__init__()
        self.r_s = r_s
        self.plot = plot
        self.delta = delta

    def get_krange(self):
        r"""Returns $k_s \Delta$ as defined in Eq 6 of Nishimishi 2018"""
        k_s = 2 * np.pi / self.r_s  # BAO Wavenumber
        k_range = self.delta * k_s  # Range of k to sum over
        return k_range

    def postprocess(self, ks, pk, mask, return_denominator=False, plot=False):
        """Runs the BAO Extractor method and returns the extracted BAO signal.

        Warning that this is the estimator given in Eq5 Nishimichi 2018

        As such, make sure your k values are finely sampled and linearly spaced. Alas for
        our data, this isn't always possible to do because the window function wrecks us.

        Parameters
        ----------
        ks : np.array
            The k values for the BAO power spectrum
        pk : np.array
            The power spectrum at `ks`

        Returns
        -------

        """
        k_range = self.get_krange()

        result = []
        denoms = []
        for k, p in zip(ks, pk):
            k_diff = np.abs(ks - k)
            m = k_diff < k_range
            numerator = (1 - (pk[m] / p)).sum()
            denominator = (1 - np.cos(self.r_s * (ks[m] - k))).sum()
            res = numerator / denominator
            denoms.append(denominator)
            result.append(res)
        result = np.array(result)

        if mask is None:
            mask = np.ones(result.shape).astype(np.bool)

        # Plots for debugging purposes to make sure everything looks good
        if self.plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(nrows=2, figsize=(5, 7))
            axes[0].plot(ks[mask], ks[mask] * pk[mask], label="Input")
            axes[1].plot(ks[mask], result[mask], label="Output")
            axes[0].set_ylabel(r"$k\,P(k)$")
            axes[0].set_title("Extracted BA0")
            axes[1].set_ylabel(r"$R(k)$")
            axes[1].set_xlabel(r"$k$")
            plt.show()

        # Optionally return the denominator instead
        # Used for manually verifying the correctness of the covariance
        # described in Eq7 (and Noda2019 eq 21,22,23)
        if return_denominator:
            return np.array(denoms)[mask]
        return result[mask]


class BAOExtractor(PureBAOExtractor):
    """Implements the mix of BAO extractor and power spectrum as defined in Noda 2019, with
    index mixing taken from page 9, paragraph 1 and confirmed via private communication:

    pi_i = {1, 2, 3, 7, 15}
    rho_i = {4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25}

    """

    def __init__(self, r_s, plot=False, delta=0.6, mink=0.06, extra_ks=(0.0925, 0.1775), reorder=True, invert=False):
        super().__init__(r_s, plot=plot, delta=delta)
        self.mink = mink
        self.extra_ks = extra_ks
        self.reorder = reorder
        self.invert = invert

    def get_is_extracted(self, ks):
        # Use indexes to blend the two together
        indices = np.array(list(range(ks.size)))
        extra = None
        for e in self.extra_ks:
            ind = np.argmin(np.abs(ks - e))
            if extra is None:
                extra = indices == ind
            else:
                extra |= indices == ind
        mask_power = (ks < self.mink) | extra
        if self.invert:
            return mask_power
        else:
            return ~mask_power

    def postprocess(self, ks, pk, mask):
        """Process the power spectrum to get a mix of extracted BAO and P(k)

        Parameters
        ----------
        ks : np.ndarray
            Wavenumbers
        pk : np.ndarray
            Power at wavenumber
        mask : np.ndarray (bool mask), optional
            Which k values to return at the end. Used to remove k values below / above certain values.
            I pass them in here because if we reorder the k values the masking cannot be done outside this function.
        """
        if mask is None:
            mask = np.ones(pk.shape).astype(np.bool)
        extracted_pk = super().postprocess(ks, pk, None)
        mask_bao = self.get_is_extracted(ks)
        if self.reorder:
            result = np.concatenate((pk[mask & ~mask_bao], extracted_pk[mask & mask_bao]))
        else:
            mask_int = mask_bao.astype(np.int)
            result = (extracted_pk * (mask_int) + pk * (1 - mask_int))[mask]
        return result


if __name__ == "__main__":

    import sys
    import matplotlib.pyplot as plt
    from scipy.interpolate import splev, splrep
    from barry.cosmology.camb_generator import getCambGenerator

    sys.path.append("../..")
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Apply the BAO extractor to the CAMB power spectrum and plot
    c = getCambGenerator()
    ks = c.ks
    r_s, pk_lin = c.get_data()["r_s"], c.get_data()["pk_lin_z"]

    # Bin similar to how we would bin for real data
    ks2 = np.linspace(0.0, 0.4, 50)
    pk_lin2 = splev(ks2, splrep(ks, pk_lin))

    # The plot is included in the PureBAOExtractor class
    b = PureBAOExtractor(r_s, plot=True)
    pk_extract = b.postprocess(ks2, pk_lin2, mask=np.where(np.logical_and(ks2 > 0.02, ks2 < 0.3)))
