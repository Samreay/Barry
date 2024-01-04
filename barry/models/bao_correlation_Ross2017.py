import sys

sys.path.append("../..")
from barry.models import PowerBeutler2017
from barry.models.bao_correlation import CorrelationFunctionFit
from scipy.interpolate import splev, splrep
import numpy as np


class CorrRoss2017(CorrelationFunctionFit):
    """xi(s) model inspired from Beutler 2017 and Ross 2017."""

    def __init__(
        self,
        name="Corr Ross 2017",
        fix_params=("om",),
        smooth_type=None,
        recon=None,
        smooth=False,
        correction=None,
        isotropic=False,
        poly_poles=(0, 2),
        marg=None,
        includeb2=True,
        include_binmat=True,
        broadband_type="spline",
        **kwargs,
    ):

        super().__init__(
            name=name,
            fix_params=fix_params,
            smooth_type=smooth_type,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            poly_poles=poly_poles,
            marg=marg,
            includeb2=includeb2,
            include_binmat=include_binmat,
            broadband_type=broadband_type,
            **kwargs,
        )
        self.parent = PowerBeutler2017(
            fix_params=fix_params,
            smooth_type=smooth_type,
            recon=recon,
            smooth=smooth,
            correction=correction,
            isotropic=isotropic,
            marg=marg,
            broadband_type=None,
        )

        if self.includeb2:
            if 4 in poly_poles:
                marg_bias = 3
            else:
                marg_bias = 2
        else:
            marg_bias = 1
        self.set_marg(fix_params, do_bias=False, marg_bias=0)

        self.fixed_xi = False
        self.store_xi = [None, None, None]
        self.store_xi_smooth = [None, None, None]

        # We might not need to evaluate the hankel transform everytime, so check.
        if self.isotropic:
            if all(elem in self.fix_params for elem in ["sigma_s", "sigma_nl"]):
                self.fixed_xi = True
        else:
            if all(elem in self.fix_params for elem in ["sigma_s", "sigma_nl_perp", "sigma_nl_par", "beta"]):
                self.fixed_xi = True

    def declare_parameters(self):
        super().declare_parameters()
        self.add_param("sigma_s", r"$\Sigma_s$", 0.0, 20.0, 10.0)  # Fingers-of-god damping
        if self.isotropic:
            self.add_param("sigma_nl", r"$\Sigma_{nl}$", 0.0, 20.0, 10.0)  # BAO damping
        else:
            self.add_param("beta", r"$\beta$", 0.01, 4.0, None)  # RSD parameter f/b
            self.add_param("sigma_nl_par", r"$\Sigma_{nl,||}$", 0.0, 20.0, 8.0)  # BAO damping parallel to LOS
            self.add_param("sigma_nl_perp", r"$\Sigma_{nl,\perp}$", 0.0, 20.0, 4.0)  # BAO damping perpendicular to LOS
        for pole in self.poly_poles:
            if self.includeb2:
                if pole != 0:
                    self.add_param(f"b{{{pole}}}_{{{1}}}", f"$b_{{{pole}_{1}}}$", 0.01, 10.0, 1.0)  # Linear galaxy bias for each multipole

    def compute_correlation_function(self, dist, p, smooth=False):
        """Computes the correlation function model using the Beutler et. al., 2017 power spectrum
            and 3 bias parameters and polynomial terms per multipole

        Parameters
        ----------
        dist : np.ndarray
            Array of distances in the correlation function to compute
        p : dict
            dictionary of parameter name to float value pairs
        smooth : bool, optional
            Whether or not to generate a smooth model without the BAO feature

        Returns
        -------
        sprime : np.ndarray
            distances of the computed xi
        xi : np.ndarray
            the model monopole, quadrupole and hexadecapole interpolated to sprime.
        poly: np.ndarray
            the additive terms in the model, necessary for analytical marginalisation

        """

        ks = self.parent.camb.ks
        if self.fixed_xi:
            if smooth:
                if self.store_xi_smooth[0] is None:
                    _, pks = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)
            else:
                if self.store_xi[0] is None:
                    _, pks = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)
        else:
            _, pks = self.parent.compute_power_spectrum(self.parent.camb.ks, p, smooth=smooth, for_corr=True)

        xi = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]

        finedist = np.linspace(0.0, 300.0, 601)
        if self.isotropic:
            sprime = p["alpha"] * dist
            if self.fixed_xi:
                if smooth:
                    if self.store_xi_smooth[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        self.store_xi_smooth[0] = pk2xi0
                    else:
                        pk2xi0 = self.store_xi_smooth[0]
                else:
                    if self.store_xi[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        self.store_xi[0] = pk2xi0
                    else:
                        pk2xi0 = self.store_xi[0]
            else:
                pk2xi0 = self.pk2xi_0.__call__(ks, pks[0], sprime)
            xi[0] = splev(sprime, pk2xi0)
        else:
            # Construct the dilated 2D correlation function by splining the undilated multipoles. We could have computed these
            # directly at sprime, but sprime depends on both s and mu, so splining is quicker
            # epsilon = np.round(p["epsilon"], decimals=5)
            epsilon = p["epsilon"]
            sprime = np.outer(dist * p["alpha"], self.get_sprimefac(epsilon))
            muprime = self.get_muprime(epsilon)

            if self.fixed_xi:
                if smooth:
                    if self.store_xi_smooth[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                        pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))
                        self.store_xi_smooth = [pk2xi0, pk2xi2, pk2xi4]
                    else:
                        pk2xi0, pk2xi2, pk2xi4 = self.store_xi_smooth
                else:
                    if self.store_xi[0] is None:
                        pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                        pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                        pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))
                        self.store_xi = [pk2xi0, pk2xi2, pk2xi4]
                    else:
                        pk2xi0, pk2xi2, pk2xi4 = self.store_xi
            else:
                pk2xi0 = splrep(finedist, self.pk2xi_0.__call__(ks, pks[0], finedist))
                pk2xi2 = splrep(finedist, self.pk2xi_2.__call__(ks, pks[2], finedist))
                pk2xi4 = splrep(finedist, self.pk2xi_4.__call__(ks, pks[4], finedist))

            xi0 = splev(sprime, pk2xi0)
            xi2 = splev(sprime, pk2xi2)
            xi4 = splev(sprime, pk2xi4)

            xi2d = xi0 + 0.5 * (3.0 * muprime**2 - 1) * xi2 + 0.125 * (35.0 * muprime**4 - 30.0 * muprime**2 + 3.0) * xi4

            # Now compute the dilated xi multipoles, with the volume scaling as necessary
            xi[0], xi[1], xi[2] = self.integrate_mu(xi2d * p["alpha"] ** 3, self.mu)

        return sprime, xi

    def get_model(self, p, d, smooth=False):
        """Gets the model prediction using the data passed in and parameter location specified

        Parameters
        ----------
        p : dict
            A dictionary of parameter names to parameter values
        d : dict
            A specific set of data to compute the model for. For correlation functions, this needs to
            have a key of 'dist' which contains the Mpc/h value of distances to compute.
        smooth : bool, optional
            Whether to only generate a smooth model without the BAO feature

        Returns
        -------
        xi_model : np.ndarray
            The concatenated xi_{\\ell}(s) predictions at the dilated distances given p and data['dist']
        poly_model : np.ndarray
            the functions describing any polynomial terms, used for analytical marginalisation
            k values correspond to d['dist']
        """

        # For this model, the xis returned are the model components as functions of mu
        # that needs multiplying together in the appropriate way to give the multipoles
        dist, xi_comp = self.compute_correlation_function(d["dist_input"] if self.include_binmat else d["dist"], p, smooth=smooth)

        # If we are not analytically marginalising, we multiply by the galaxy bia terms and add the polynomials, then convolve with the binning matrix
        # Otherwise, we convolve the components with the binnin matrix, then build up the analytically marginalised parts
        if self.marg:

            # Load the polynomial terms if computed already, or compute them for the first time.
            if self.winpoly is None:
                if self.poly is None:
                    self.compute_poly(d["dist_input"] if self.include_binmat else d["dist"])

                nmarg, nell = np.shape(self.poly)[:2]
                len_poly = len(d["dist"]) if self.isotropic else len(d["dist"]) * nell
                self.winpoly = np.zeros((nmarg, len_poly))
                for n in range(nmarg):
                    self.winpoly[n] = np.concatenate([pol @ d["binmat"] if self.include_binmat else pol for pol in self.poly[n]])

            if self.marg_bias:

                # Convolve the xi model with the binning matrix and concatenate into a single data vector
                xi_generated = [xi @ d["binmat"] if self.include_binmat else xi for xi in xi_comp]

                # We need to update/include the poly terms corresponding to the galaxy bias
                if self.includeb2:
                    poly_b = [
                        np.concatenate([xi_generated[0], -2.5 * xi_generated[0], 1.125 * 3.0 * xi_generated[0]]),
                        np.concatenate([np.zeros(len(d["dist"])), 2.5 * xi_generated[1], -1.125 * 10.0 * xi_generated[1]]),
                    ]
                    if 4 in self.poly_poles:
                        poly_b.append(np.concatenate([np.zeros(len(d["dist"])), np.zeros(len(d["dist"])), 1.125 * xi_generated[2]]))
                else:
                    poly_b = np.concatenate(
                        [
                            xi_generated[0],
                            2.5 * (xi_generated[1] - xi_generated[0]),
                            1.125 * (xi_generated[2] - 10.0 * xi_generated[1] + 3.0 * xi_generated[0]),
                        ]
                    )
                poly_model = np.vstack([poly_b, self.winpoly])
                xi_model = np.zeros(np.shape(self.winpoly)[1])
            else:
                xis = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
                xis[0] = p["b{0}_{1}"] * xi_comp[0]
                if self.includeb2:
                    xis[1] = 2.5 * (p["b{2}_{1}"] * xi_comp[1] - xis[0])
                    if 4 in self.poly_poles:
                        xis[2] = 1.125 * (p["b{4}_{1}"] * xi_comp[2] - 10.0 * p["b{2}_{1}"] * xi_comp[1] + 3.0 * xis[0])
                    else:
                        xis[2] = 1.125 * (xi_comp[2] - 10.0 * p["b{2}_{1}"] * xi_comp[1] + 3.0 * xis[0])
                else:
                    xis[1] = 2.5 * p["b{0}_{1}"] * (xi_comp[1] - xi_comp[0])
                    xis[2] = 1.125 * p["b{0}_{1}"] * (xi_comp[2] - 10.0 * xi_comp[1] + 3.0 * xi_comp[0])

                xi_generated = [xi @ d["binmat"] if self.include_binmat else xi for xi in xis]

                poly_model = self.winpoly
                xi_model = np.concatenate([xi_generated[l] for l in range(len(d["poles"]))])

        else:
            xis = [np.zeros(len(dist)), np.zeros(len(dist)), np.zeros(len(dist))]
            xis[0] = p["b{0}_{1}"] * xi_comp[0]
            if self.includeb2:
                xis[1] = 2.5 * (p["b{2}_{1}"] * xi_comp[1] - xis[0])
                if 4 in self.poly_poles:
                    xis[2] = 1.125 * (p["b{4}_{1}"] * xi_comp[2] - 10.0 * p["b{2}_{1}"] * xi_comp[1] + 3.0 * xis[0])
                else:
                    xis[2] = 1.125 * (xi_comp[2] - 10.0 * p["b{2}_{1}"] * xi_comp[1] + 3.0 * xis[0])
            else:
                xis[1] = 2.5 * p["b{0}_{1}"] * (xi_comp[1] - xi_comp[0])
                xis[2] = 1.125 * p["b{0}_{1}"] * (xi_comp[2] - 10.0 * xi_comp[1] + 3.0 * xi_comp[0])

            if self.poly is None:
                self.compute_poly(d["dist_input"] if self.include_binmat else d["dist"])
            for pole in self.poly_poles:
                for i, ip in enumerate(self.n_poly):
                    xis[int(pole / 2)] += p[f"a{{{pole}}}_{{{ip}}}_{{{1}}}"] * self.poly[ip, int(pole / 2), :]

            # Convolve the xi model with the binning matrix and concatenate into a single data vector
            xi_generated = [xi @ d["binmat"] if self.include_binmat else xi for xi in xis]
            xi_model = np.concatenate([xi_generated[l] for l in range(len(d["poles"]))])

            poly_model = None

        return xi_model, poly_model


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from barry.datasets.dataset_correlation_function import (
        CorrelationFunction_DESI_KP4,
    )
    from barry.config import setup_logging
    from barry.models.model import Correction

    setup_logging()

    """print("Checking isotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=True, recon="iso", realisation="data")
    model = CorrBeutler2017(recon=dataset.recon, marg="full", isotropic=dataset.isotropic, correction=Correction.NONE)
    model.sanity_check(dataset)

    print("Checking anisotropic data")
    dataset = CorrelationFunction_ROSS_DR12(isotropic=False, recon="iso", fit_poles=[0, 2], realisation="data")
    model = CorrBeutler2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om"],
        poly_poles=[0, 2],
        correction=Correction.NONE,
    )
    model.sanity_check(dataset)"""

    print("Checking anisotropic data")
    dataset = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=52.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
    )

    model = CorrRoss2017(
        recon=dataset.recon,
        isotropic=dataset.isotropic,
        marg="full",
        fix_params=["om", "beta"],
        poly_poles=dataset.fit_poles,
        correction=Correction.HARTLAP,
        includeb2=True,
        n_poly=[0, 2],
    )
    model.set_default("beta", 0.4)
    model.set_default("sigma_nl_par", 5.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=1.0, prior="gaussian")
    model.set_default("sigma_s", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../../barry/data/desi_kp4/DESI_Pk_template.dat")
    model.parent.kvals, model.parent.pksmooth, model.parent.pkratio = pktemplate.T

    model.sanity_check(dataset)
