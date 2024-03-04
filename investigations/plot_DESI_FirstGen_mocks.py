import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.models.model import Correction
from barry.datasets import PowerSpectrum_DESI_KP4, CorrelationFunction_DESI_KP4

if __name__ == "__main__":

    # Load in the data to be fit
    dataset_pk = PowerSpectrum_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_k=0.02,
        max_k=0.30,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_pk_lrg.pkl",
    )
    data_pk = dataset_pk.get_data()
    kth = data_pk[0]["ks"]

    dataset_xi = CorrelationFunction_DESI_KP4(
        recon="sym",
        fit_poles=[0, 2],
        min_dist=50.0,
        max_dist=150.0,
        realisation=None,
        num_mocks=1000,
        reduce_cov_factor=25,
        datafile="desi_kp4_abacus_cubicbox_cv_xi_lrg.pkl",
    )
    data_xi = dataset_xi.get_data()
    sth = data_xi[0]["dist"]

    # Set up some fiducial models and the bestfit
    model_pk = PowerBeutler2017(
        recon=dataset_pk.recon,
        isotropic=dataset_pk.isotropic,
        fix_params=["om"],
        marg="full",
        poly_poles=dataset_pk.fit_poles,
        correction=Correction.HARTLAP,
    )
    model_pk.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
    model_pk.set_default("beta", 0.4, min=0.1, max=0.7)
    model_pk.set_default("sigma_nl_par", 5.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_pk.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=1.0, prior="gaussian")
    model_pk.set_default("sigma_s", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_pk.kvals, model_pk.pksmooth, model_pk.pkratio = pktemplate.T

    model_xi = CorrBeutler2017(
        recon=dataset_xi.recon,
        isotropic=dataset_xi.isotropic,
        fix_params=["om"],
        marg="full",
        poly_poles=dataset_xi.fit_poles,
        correction=Correction.HARTLAP,
    )
    model_xi.set_default(f"b{{{0}}}_{{{1}}}", 2.0, min=0.5, max=9.0)
    model_xi.set_default("beta", 0.4, min=0.1, max=0.7)
    model_xi.set_default("sigma_nl_par", 5.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")
    model_xi.set_default("sigma_nl_perp", 2.0, min=0.0, max=20.0, sigma=1.0, prior="gaussian")
    model_xi.set_default("sigma_s", 2.0, min=0.0, max=20.0, sigma=2.0, prior="gaussian")

    # Load in a pre-existing BAO template
    pktemplate = np.loadtxt("../barry/data/desi_kp4/DESI_Pk_template.dat")
    model_xi.parent.kvals, model_xi.parent.pksmooth, model_xi.parent.pkratio = pktemplate.T

    params_pk = {
        "b{0}_{1}": 1.9215056051818582,
        "alpha": 0.999690639650866,
        "epsilon": 0.0008001023628552706,
        "sigma_s": 1.9978412970738453,
        "beta": 0.3457777530990704,
        "sigma_nl_par": 4.962751087634123,
        "sigma_nl_perp": 1.8844308623904393,
    }
    params_pk.update({(p.name, p.default) for p in model_pk.get_inactive_params()})
    params_xi = {
        "b{0}_{1}": 1.8711800474460003,
        "alpha": 1.0003826852581044,
        "epsilon": 7.914924694021885e-05,
        "sigma_s": 1.9853487972252049,
        "beta": 0.34901548921329706,
        "sigma_nl_par": 4.943782363130673,
        "sigma_nl_perp": 1.9100773871359968,
    }
    params_xi.update({(p.name, p.default) for p in model_xi.get_inactive_params()})
    if not params_pk:
        model_pk.sanity_check(dataset_pk)
    else:
        model_pk.set_data(dataset_pk.get_data())
    if not params_xi:
        model_xi.sanity_check(dataset_xi)
    else:
        model_xi.set_data(dataset_xi.get_data())
    new_chi_squared_pk, dof_pk, bband_pk, mods_pk, smooths_pk = model_pk.get_model_summary(params_pk)
    new_chi_squared_xi, dof_xi, bband_xi, mods_xi, smooths_xi = model_xi.get_model_summary(params_xi)

    # Split up Pk and Xi
    fig = plt.figure(figsize=(12, 5.5))
    axes = gridspec.GridSpec(1, 2, figure=fig, left=0.1, top=0.95, bottom=0.1, right=0.95, hspace=0.0, wspace=0.2, width_ratios=[1.75, 1])
    subaxes = axes[0, 0].subgridspec(2, 1, hspace=0.05, wspace=0.0)
    ax1 = fig.add_subplot(subaxes[0, 0])
    ax2 = fig.add_subplot(subaxes[1, 0])
    subaxes = axes[0, 1].subgridspec(2, 1, hspace=0.05, wspace=0.0)
    ax3 = fig.add_subplot(subaxes[0, 0])
    ax4 = fig.add_subplot(subaxes[1, 0])

    c, idata = "#1f77b4", 0
    errs_pk = np.sqrt(np.diag(data_pk[0]["cov"])).reshape((-1, data_pk[0]["ndata"], len(data_pk[0]["ks"])))[::2]
    ax1.errorbar(
        data_pk[0]["ks"],
        data_pk[0]["ks"] * data_pk[0]["pk0"][idata],
        yerr=data_pk[0]["ks"] * errs_pk[0][idata],
        fmt="o",
        mfc=c,
        label="$P_{0}(k)$",
        c=c,
    )
    ax1.errorbar(
        data_pk[0]["ks"],
        data_pk[0]["ks"] * data_pk[0]["pk2"][idata],
        yerr=data_pk[0]["ks"] * errs_pk[1][idata],
        fmt="o",
        mfc="w",
        label="$P_{2}(k)$",
        c=c,
    )
    ax1.plot(kth, kth * mods_pk[0][idata], c=c)
    ax1.plot(kth, kth * mods_pk[2][idata], c=c)
    ax1.plot(kth, kth * smooths_pk[0][idata], c=c, ls="--")
    ax1.plot(kth, kth * smooths_pk[2][idata], c=c, ls="--")
    ax1.set_ylabel("$k\,P_{\ell}(k)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax1.set_xticklabels([])

    ax2.errorbar(
        data_pk[0]["ks"],
        data_pk[0]["ks"] * (data_pk[0]["pk0"][idata] - smooths_pk[0][idata]) + 100.0,
        yerr=data_pk[0]["ks"] * errs_pk[0][idata],
        fmt="o",
        mfc=c,
        c=c,
    )
    ax2.errorbar(
        data_pk[0]["ks"],
        data_pk[0]["ks"] * (data_pk[0]["pk2"][idata] - smooths_pk[2][idata]) - 100.0,
        yerr=data_pk[0]["ks"] * errs_pk[1][idata],
        fmt="o",
        mfc="w",
        c=c,
    )

    ax2.axhline(100.0, c=c, ls="--")
    ax2.axhline(-100.0, c=c, ls="--")
    ax2.plot(kth, kth * (mods_pk[0][idata] - smooths_pk[0][idata]) + 100.0, c=c)
    ax2.plot(kth, kth * (mods_pk[2][idata] - smooths_pk[2][idata]) - 100.0, c=c)
    ax2.set_xlabel("$k\quad[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
    ax2.set_ylabel("$k\,\mathcal{C}_{\ell}(k)\,P_{w,\ell}(k)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax2.legend(
        [
            plt.errorbar([], [], fmt="o", c="k"),
            plt.errorbar([], [], marker=None, c="k", ls="-"),
            plt.errorbar([], [], marker=None, c="k", ls="--"),
        ],
        [r"$\mathrm{Data}$", r"$\mathrm{Full\,Model}$", r"$\mathrm{No-Wiggle\,Model}$"],
        ncol=3,
        fontsize=10,
        loc="upper right",
    )

    # Correlation function. Need to regenerate the model as unlike the power spectrum we only evaluated this over the narrower fitting range
    # We use the old bband parameters though as we want to use the ones based on the fiducial fitting range
    c = "#ff7f0e"
    errs_xi = np.sqrt(np.diag(data_xi[0]["cov"])).reshape((-1, len(data_xi[0]["dist"])))
    ax3.errorbar(
        data_xi[0]["dist"],
        data_xi[0]["dist"] ** 2 * data_xi[0]["xi0"],
        yerr=data_xi[0]["dist"] ** 2 * errs_xi[0],
        fmt="o",
        mfc=c,
        label="$\\xi_{0}(s)$",
        c=c,
    )
    ax3.errorbar(
        data_xi[0]["dist"],
        data_xi[0]["dist"] ** 2 * data_xi[0]["xi2"],
        yerr=data_xi[0]["dist"] ** 2 * errs_xi[1],
        fmt="o",
        mfc="w",
        label="$\\xi_{2}(s)$",
        c=c,
    )
    ax3.plot(sth, sth**2 * mods_xi[0], c=c)
    ax3.plot(sth, sth**2 * mods_xi[1], c=c)
    ax3.plot(sth, sth**2 * smooths_xi[0], c=c, ls="--")
    ax3.plot(sth, sth**2 * smooths_xi[1], c=c, ls="--")
    ax3.set_ylabel(r"$s^{2}\,\xi_{\ell}(s)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax3.set_xticklabels([])

    ax4.errorbar(
        data_xi[0]["dist"],
        data_xi[0]["dist"] ** 2 * (data_xi[0]["xi0"] - smooths_xi[0]) + 10.0,
        yerr=data_xi[0]["dist"] ** 2 * errs_xi[0],
        fmt="o",
        mfc=c,
        label="$\\xi_{0}(s)$",
        c=c,
    )
    ax4.errorbar(
        data_xi[0]["dist"],
        data_xi[0]["dist"] ** 2 * (data_xi[0]["xi2"] - smooths_xi[1]) - 10.0,
        yerr=data_xi[0]["dist"] ** 2 * errs_xi[1],
        fmt="o",
        mfc="w",
        label="$\\xi_{2}(s)$",
        c=c,
    )
    ax4.axhline(10.0, c=c, ls="--")
    ax4.axhline(-10.0, c=c, ls="--")
    ax4.plot(sth, sth**2 * (mods_xi[0] - smooths_xi[0]) + 10.0, c=c)
    ax4.plot(sth, sth**2 * (mods_xi[1] - smooths_xi[1]) - 10.0, c=c)
    ax4.set_xlabel("$s\quad[h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax4.set_ylabel(r"$s^{2}\,\mathcal{C}_{\ell}(s)\,\xi_{w,\ell}(s)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax4.legend(
        [plt.errorbar([], [], fmt="o", c="k"), plt.errorbar([], [], fmt="o", mfc="w", c="k")],
        [r"$\ell=0$", r"$\ell=2$"],
        ncol=1,
        fontsize=10,
        loc="upper left",
    )

    plt.show()

    # Output the data for plotting purposes
    np.savetxt(
        "./ChenHowlettPaperPlots/Figure7_pk.txt",
        np.c_[
            data_pk[0]["ks"],
            data_pk[0]["pk0"][idata],
            data_pk[0]["pk2"][idata],
            mods_pk[0][idata],
            mods_pk[2][idata],
            smooths_pk[0][idata],
            smooths_pk[2][idata],
            errs_pk[0][idata],
            errs_pk[1][idata],
        ],
        header="k, pk_0, pk_2, model_pk_0, model_pk_2, smoothmodel_pk_0, smoothmodel_pk_2, err_pk_0, err_pk_2",
    )
    np.savetxt(
        "./ChenHowlettPaperPlots/Figure7_xi.txt",
        np.c_[
            data_xi[0]["dist"],
            data_xi[0]["xi0"],
            data_xi[0]["xi2"],
            mods_xi[0],
            mods_xi[1],
            smooths_xi[0],
            smooths_xi[1],
            errs_xi[0],
            errs_xi[1],
        ],
        header="s, xi_0, xi_2, model_xi_0, model_xi_2, smoothmodel_xi_0, smoothmodel_xi_2, err_xi_0, err_xi_2",
    )
