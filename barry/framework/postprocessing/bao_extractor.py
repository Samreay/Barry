import numpy as np
from barry.framework.postprocessing.postprocessor import PkPostProcess


class PureBAOExtractor(PkPostProcess):
    """
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

    def postprocess(self, ks, pk, return_denominator=False):
        """ Runs the BAO Extractor method and returns the extracted BAO signal.

        Warning that this is the estimator given in Eq5 Nishimichi et al 2018 (1708.00375)

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
        k_s = 2 * np.pi / self.r_s  # BAO Wavenumber
        k_range = self.delta * k_s  # Range of k to sum over

        result = []
        denoms = []
        for k, p in zip(ks, pk):
            k_diff = np.abs(ks - k)
            mask = k_diff < k_range
            numerator = (1 - (pk[mask] / p)).sum()
            denominator = (1 - np.cos(self.r_s * (ks[mask] - k))).sum()
            res = numerator / denominator
            denoms.append(denominator)
            result.append(res)

        if self.plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=2, figsize=(5, 7))
            axes[0].plot(ks, pk, label="Input")
            axes[1].plot(ks, result, label="Output")
            plt.show()

        if return_denominator:
            return np.array(denoms)
        return np.array(result)


class BAOExtractor(PureBAOExtractor):
    def __init__(self, r_s, plot=False, delta=0.6, mink=0.05, maxk=0.15):
        super().__init__(r_s, plot=plot, delta=delta)
        self.mink = mink
        self.maxk = maxk

    def postprocess(self, ks, pk):
        extracted_pk = super().postprocess(ks, pk)

        # Use indexes to blend the two together
        indices = np.array(list(range(ks.size)))
        mask_bao = ((ks < self.mink) | (indices % 2 == 1)) & (ks < self.maxk)
        result = np.concatenate((extracted_pk[~mask_bao], pk[mask_bao]))
        return result


if __name__ == "__main__":
    from barry.framework.cosmology.camb_generator import CambGenerator

    camb = CambGenerator(om_resolution=10, h0_resolution=1)
    ks = camb.ks
    print(ks.shape)
    r_s, pk_lin = camb.get_data(0.3, 0.70)

    from scipy.interpolate import splev, splrep
    rep = splrep(ks, pk_lin)
    # ks2 = np.linspace(ks.min(), 1, 1000)
    ks2 = np.linspace(0, 0.398, 100) # Matching the winfit_2 data binning
    pk_lin2 = splev(ks2, rep)

    print("Got pklin")
    b = BAOExtractor(r_s)
    pk_extract = b.postprocess(ks2, pk_lin2)
    print("Got pk_extract")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=2, figsize=(5, 9), sharex=True)
    axes[0].plot(ks2, pk_lin2)
    axes[0].set_title("pk_lin")
    axes[1].plot(ks2, pk_extract)
    axes[1].set_title("Extracted BAO, using winfit_2 bins (0, 0.398, 100)")
    plt.show()

    from barry.framework.datasets.mock_power import MockPowerSpectrum
    dataset = MockPowerSpectrum(name="Recon mean", recon=True, min_k=0.02, step_size=2, postprocess=b)
    data = dataset.get_data()
    import seaborn as sb
    sb.heatmap(data["corr"])
    plt.show()
