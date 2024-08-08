import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig


def Figure7():

    data_pk = pd.read_csv("./Figure7_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    data_xi = pd.read_csv("./Figure7_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T

    # Split up Pk and Xi
    fig = plt.figure(figsize=(12, 5.5))
    axes = gridspec.GridSpec(1, 2, figure=fig, left=0.1, top=0.95, bottom=0.1, right=0.95, hspace=0.0, wspace=0.2, width_ratios=[1.75, 1])
    subaxes = axes[0, 0].subgridspec(2, 1, hspace=0.05, wspace=0.0)
    ax1 = fig.add_subplot(subaxes[0, 0])
    ax2 = fig.add_subplot(subaxes[1, 0])
    subaxes = axes[0, 1].subgridspec(2, 1, hspace=0.05, wspace=0.0)
    ax3 = fig.add_subplot(subaxes[0, 0])
    ax4 = fig.add_subplot(subaxes[1, 0])

    # Power spectrum plot
    c = "#1f77b4"
    ax1.errorbar(
        data_pk[0],
        data_pk[0] * data_pk[1],
        yerr=data_pk[0] * data_pk[7],
        fmt="o",
        mfc=c,
        label=r"$P_{0}(k)$",
        c=c,
    )
    ax1.errorbar(
        data_pk[0],
        data_pk[0] * data_pk[2],
        yerr=data_pk[0] * data_pk[8],
        fmt="o",
        mfc="w",
        label=r"$P_{2}(k)$",
        c=c,
    )
    ax1.plot(data_pk[0], data_pk[0] * data_pk[3], c="#1f77b4")
    ax1.plot(data_pk[0], data_pk[0] * data_pk[4], c="#1f77b4")
    ax1.plot(data_pk[0], data_pk[0] * data_pk[5], c="#1f77b4", ls="--")
    ax1.plot(data_pk[0], data_pk[0] * data_pk[6], c="#1f77b4", ls="--")
    ax1.set_ylabel(r"$k\,P_{\ell}(k)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax1.set_xticklabels([])

    ax2.errorbar(
        data_pk[0],
        data_pk[0] * (data_pk[1] - data_pk[5]) + 100.0,
        yerr=data_pk[0] * data_pk[7],
        fmt="o",
        mfc=c,
        c=c,
    )
    ax2.errorbar(
        data_pk[0],
        data_pk[0] * (data_pk[2] - data_pk[6]) - 100.0,
        yerr=data_pk[0] * data_pk[8],
        fmt="o",
        mfc="w",
        c=c,
    )
    ax2.axhline(100.0, c=c, ls="--")
    ax2.axhline(-100.0, c=c, ls="--")
    ax2.plot(data_pk[0], data_pk[0] * (data_pk[3] - data_pk[5]) + 100.0, c=c)
    ax2.plot(data_pk[0], data_pk[0] * (data_pk[4] - data_pk[6]) - 100.0, c=c)
    ax2.set_xlabel(r"$k\quad[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
    ax2.set_ylabel(r"$k\,\mathcal{C}_{\ell}(k)\,P_{w,\ell}(k)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
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

    # Correlation function plot
    c = "#ff7f0e"
    ax3.errorbar(
        data_xi[0],
        data_xi[0] ** 2 * data_xi[1],
        yerr=data_xi[0] ** 2 * data_xi[7],
        fmt="o",
        mfc=c,
        label=r"$\xi_{0}(s)$",
        c=c,
    )
    ax3.errorbar(
        data_xi[0],
        data_xi[0] ** 2 * data_xi[2],
        yerr=data_xi[0] ** 2 * data_xi[8],
        fmt="o",
        mfc="w",
        label=r"$\xi_{2}(s)$",
        c=c,
    )
    ax3.plot(data_xi[0], data_xi[0] ** 2 * data_xi[3], c=c)
    ax3.plot(data_xi[0], data_xi[0] ** 2 * data_xi[4], c=c)
    ax3.plot(data_xi[0], data_xi[0] ** 2 * data_xi[5], c=c, ls="--")
    ax3.plot(data_xi[0], data_xi[0] ** 2 * data_xi[6], c=c, ls="--")
    ax3.set_ylabel(r"$s^{2}\,\xi_{\ell}(s)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax3.set_xticklabels([])

    ax4.errorbar(
        data_xi[0],
        data_xi[0] ** 2 * (data_xi[1] - data_xi[5]) + 10.0,
        yerr=data_xi[0] ** 2 * data_xi[7],
        fmt="o",
        mfc=c,
        label=r"$\xi_{0}(s)$",
        c=c,
    )
    ax4.errorbar(
        data_xi[0],
        data_xi[0] ** 2 * (data_xi[2] - data_xi[6]) - 10.0,
        yerr=data_xi[0] ** 2 * data_xi[8],
        fmt="o",
        mfc="w",
        label=r"$\xi_{2}(s)$",
        c=c,
    )
    ax4.axhline(10.0, c=c, ls="--")
    ax4.axhline(-10.0, c=c, ls="--")
    ax4.plot(data_xi[0], data_xi[0] ** 2 * (data_xi[3] - data_xi[5]) + 10.0, c=c)
    ax4.plot(data_xi[0], data_xi[0] ** 2 * (data_xi[4] - data_xi[6]) - 10.0, c=c)
    ax4.set_xlabel(r"$s\quad[h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax4.set_xlabel(r"$s\quad[h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
    ax4.set_ylabel(r"$s^{2}\,\mathcal{C}_{\ell}(s)\,\xi_{w,\ell}(s)\quad[h^{-2}\mathrm{Mpc^{2}}]$", fontsize=12)
    ax4.legend(
        [plt.errorbar([], [], fmt="o", c="k"), plt.errorbar([], [], fmt="o", mfc="w", c="k")],
        [r"$\ell=0$", r"$\ell=2$"],
        ncol=1,
        fontsize=10,
        loc="upper left",
    )

    plt.show()


def Figure11(row="top"):

    if row == "top":
        stats_pk = pd.read_csv("./Figure11_sigma_par_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        stats_xi = pd.read_csv("./Figure11_sigma_par_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        label = r"$\Sigma_{||}$"
        outfile = "./Figure11_top.png"
    elif row == "middle":
        stats_pk = pd.read_csv("./Figure11_sigma_perp_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        stats_xi = pd.read_csv("./Figure11_sigma_perp_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        label = r"$\Sigma_{\perp}$"
        outfile = "./Figure11_middle.png"
    else:
        stats_pk = pd.read_csv("./Figure11_sigma_s_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        stats_xi = pd.read_csv("./Figure11_sigma_s_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy()
        label = r"$\Sigma_{s}$"
        outfile = "./Figure11_bottom.png"

    # Split up Pk and Xi
    fig = plt.figure(figsize=(4, 2))
    axes = gridspec.GridSpec(2, 2, figure=fig, left=0.1, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
    for ind in range(2):
        tracer = r"$\xi(s)$" if ind == 0 else r"$P(k)$"
        dat = stats_xi if ind == 0 else stats_pk
        ax1 = fig.add_subplot(axes[0, ind])
        ax2 = fig.add_subplot(axes[1, ind])

        c = "#ff7f0e" if ind == 0 else "#1f77b4"
        ax1.plot(dat[:, 0], dat[:, 1], color=c, zorder=1, alpha=0.75, lw=0.8)
        ax2.plot(dat[:, 0], dat[:, 2], color=c, zorder=1, alpha=0.75, lw=0.8)
        ax1.fill_between(
            dat[:, 0],
            (dat[:, 1] - dat[:, 3]),
            (dat[:, 1] + dat[:, 3]),
            color=c,
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax2.fill_between(
            dat[:, 0],
            (dat[:, 2] - dat[:, 4]),
            (dat[:, 2] + dat[:, 4]),
            color=c,
            zorder=1,
            alpha=0.5,
            lw=0.8,
        )
        ax1.set_xlim(0.0, 9.2)
        ax2.set_xlim(0.0, 9.2)
        ax1.set_ylim(-0.35, 0.35)
        ax2.set_ylim(-0.95, 0.95)
        ax2.set_xlabel(label)
        if ind == 0:
            ax1.set_ylabel(r"$\Delta \alpha_{\mathrm{iso}}\,(\%)$")
            ax2.set_ylabel(r"$\Delta \alpha_{\mathrm{ap}}\,(\%)$")
        else:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        ax1.set_xticklabels([])
        for val, ls in zip([-0.1, 0.0, 0.1], [":", "--", ":"]):
            ax1.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
        for val, ls in zip([-0.2, 0.0, 0.2], [":", "--", ":"]):
            ax2.axhline(val, color="k", ls=ls, zorder=0, lw=0.8)
        ax1.axvline(5.0, color="k", ls=":", zorder=0, lw=0.8)
        ax2.axvline(5.0, color="k", ls=":", zorder=0, lw=0.8)
        ax1.text(
            0.05,
            0.95,
            tracer,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=c,
        )
    fig.savefig(outfile, bbox_inches="tight", transparent=True, dpi=300)


def Figure12():

    chain_smooth = pd.read_csv(
        "./Figure12_FoG_Smooth.txt",
        sep=r"\s+",
        header=None,
        skiprows=1,
        names=[r"$\Sigma_{||}$", r"$\Sigma_{\perp}$", r"$\Sigma_s$", r"weight"],
    )
    chain_wiggles = pd.read_csv(
        "./Figure12_FoG_Wiggles.txt",
        sep=r"\s+",
        header=None,
        skiprows=1,
        names=[r"$\Sigma_{||}$", r"$\Sigma_{\perp}$", r"$\Sigma_s$", r"weight"],
    )

    truth = {
        r"$\Sigma_{||}$": 5.0,
        r"$\Sigma_{\perp}$": 2.0,
        r"$\Sigma_s$": 2.0,
    }
    extents = {
        r"$\Sigma_{||}$": [2.0, 6.0],
        r"$\Sigma_{\perp}$": [0.0, 3.0],
        r"$\Sigma_s$": [0.0, 4.0],
    }

    c = ChainConsumer()
    c.add_chain(Chain(samples=chain_wiggles, name=r"$\xi(s)\,\mathrm{FoG\,Wiggles}$", kde=True, linewidth=1.7))
    c.add_chain(Chain(samples=chain_smooth, name=r"$\xi(s)\,\mathrm{FoG\,Smooth}$", kde=True, linewidth=1.7))
    c.add_truth(Truth(location=truth, line_width=1.7))
    c.plotter.set_config(PlotConfig(extents=extents, serif=True, label_font_size=22, tick_font_size=16, legend_kwargs={"fontsize": 16}))
    c.plotter.plot(figsize="COLUMN", filename="./Figure12.png")


def Figure13():

    val_poly_xi = pd.read_csv("./Figure13_poly_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    val_spline_xi = pd.read_csv("./Figure13_spline_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    val_poly_pk = pd.read_csv("./Figure13_poly_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    val_spline_pk = pd.read_csv("./Figure13_spline_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T

    boxprops_pk = {"lw": 1.3, "color": "#1f77b4"}
    boxprops_xi = {"lw": 1.3, "color": "#ff7f0e"}
    medianprops = {"lw": 1.5, "color": "g"}
    whiskerprops = {"lw": 1.3, "color": "k"}

    fig, axes = plt.subplots(figsize=(8, 10), nrows=4, ncols=1, sharex=True, squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.08, right=0.95, hspace=0.0, wspace=0.0)
    plt.suptitle(r"$\mathrm{FoG\,Smooth\,vs.\,Wiggle\,Differences}$", fontsize=16)
    for i, (val, bp) in enumerate(
        zip([val_poly_xi, val_poly_pk, val_spline_xi, val_spline_pk], [boxprops_xi, boxprops_pk, boxprops_xi, boxprops_pk])
    ):
        for panel in range(2):
            axes[panel, 0].boxplot(
                val[panel],
                positions=[i],
                widths=0.4,
                whis=(0, 100),
                showfliers=False,
                boxprops=bp,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                capprops=whiskerprops,
            )
            axes[panel + 2, 0].boxplot(
                val[panel] / val[panel + 2],
                positions=[i],
                widths=0.4,
                whis=(0, 100),
                showfliers=False,
                boxprops=bp,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                capprops=whiskerprops,
            )
    for panel in [0, 2, 3]:
        axes[panel, 0].axhline(y=0.0, color="k", ls="-", zorder=0, lw=1.5, alpha=0.75)
        axes[panel, 0].axhline(y=-0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
        axes[panel, 0].axhline(y=0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=0.0, color="k", ls="-", zorder=0, lw=1.5, alpha=0.75)
    axes[1, 0].axhline(y=-0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)

    axes[0, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{iso}} (\%)$", fontsize=16)
    axes[1, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{ap}} (\%)$", fontsize=16)
    axes[2, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{iso}} / \sigma_{\alpha_{\mathrm{iso}}}$", fontsize=16)
    axes[3, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{ap}} / \sigma_{\alpha_{\mathrm{ap}}}$", fontsize=16)
    axes[3, 0].set_xlabel(r"$\mathrm{Method}$", fontsize=16)
    plt.setp(
        axes,
        xticks=np.arange(4),
        xticklabels=[
            r"$\mathrm{Polynomial}\,\xi(s)$",
            r"$\mathrm{Polynomial}\,P(k)$",
            r"$\mathrm{Spline}\,\xi(s)$",
            r"$\mathrm{Spline}\,P(k)$",
        ],
    )
    plt.setp(axes[3, 0].get_xticklabels(), rotation=45, fontsize=12)
    for panel in range(4):
        axes[panel, 0].xaxis.set_tick_params(labelsize=12)
        axes[panel, 0].yaxis.set_tick_params(labelsize=12)
        for axis in ["top", "bottom", "left", "right"]:
            axes[panel, 0].spines[axis].set_linewidth(1.3)

    fig.savefig("./Figure13.png", bbox_inches="tight", transparent=True, dpi=300)


def contour_rect(data, edgeval):

    im = np.where(np.fabs(data) <= edgeval, 1, 0)

    pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding

    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

    lines = []

    for ii, jj in np.ndindex(im0.shape):
        if im0[ii, jj] == 1:
            lines += [([ii - 0.5, ii - 0.5], [jj - 0.5, jj + 0.5])]
        if im1[ii, jj] == 1:
            lines += [([ii - 0.5, ii + 0.5], [jj - 0.5, jj - 0.5])]

    return lines


def Figure14():

    stats = pd.read_csv("./Figure14.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    kmins, kmaxs = np.unique(stats[0]), np.unique(stats[1])

    dkmin = kmins[1] - kmins[0]
    dkmax = kmaxs[1] - kmaxs[0]

    bestmean = np.where((stats[0] == 0.02) & (stats[1] == 0.30))[0][0]

    # Upper panel
    fig, axes = plt.subplots(figsize=(5, 3), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.15, top=0.97, bottom=0.17, right=0.8, hspace=0.0, wspace=0.10)

    axes[0, 0].imshow(
        stats[2].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.25,
        vmax=0.25,
    )
    cax = axes[0, 1].imshow(
        stats[3].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.25,
        vmax=0.25,
    )
    lines = contour_rect(stats[2].reshape(len(kmins), len(kmaxs)).T, 0.1)
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(stats[3].reshape(len(kmins), len(kmaxs)).T, 0.2)
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    axes[0, 1].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    fig.supxlabel(r"$k_{\mathrm{min}}\,(h\,\mathrm{Mpc}^{-1})$", x=0.45)
    fig.supylabel(r"$k_{\mathrm{max}}\,(h\,\mathrm{Mpc}^{-1})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=r"$\Delta \alpha_{\mathrm{iso},\mathrm{ap}}\,(\%)$",
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(kmins[0] - 0.0055, kmins[-1] + 0.0055)
    axes[0, 0].set_ylim(kmaxs[0] - 0.0105, kmaxs[-1] + 0.011)
    axes[0, 1].set_xlim(kmins[0] - 0.0055, kmins[-1] + 0.0055)
    axes[0, 1].set_ylim(kmaxs[0] - 0.0105, kmaxs[-1] + 0.011)

    # Lower panel
    fig, axes = plt.subplots(figsize=(5, 3), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.15, top=0.97, bottom=0.17, right=0.8, hspace=0.0, wspace=0.10)

    stats[4:] /= stats[4:, bestmean][:, None]

    axes[0, 0].imshow(
        stats[4].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.50,
        vmax=1.50,
    )
    cax = axes[0, 1].imshow(
        stats[5].reshape(len(kmins), len(kmaxs)).T,
        extent=(kmins[0] - 0.005, kmins[-1] + 0.005, kmaxs[0] - 0.01, kmaxs[-1] + 0.01),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.50,
        vmax=1.50,
    )
    lines = contour_rect(stats[2].reshape(len(kmins), len(kmaxs)).T, 0.1)
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(stats[3].reshape(len(kmins), len(kmaxs)).T, 0.2)
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dkmin + kmins[0], np.array(line[0]) * dkmax + kmaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    axes[0, 1].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=14, ls="None")
    fig.supxlabel(r"$k_{\mathrm{min}}\,(h\,\mathrm{Mpc}^{-1})$", x=0.45)
    fig.supylabel(r"$k_{\mathrm{max}}\,(h\,\mathrm{Mpc}^{-1})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=(r"$\mathrm{Relative}\,\,\sigma_{\alpha_{\mathrm{iso},\mathrm{ap}}}$"),
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(kmins[0] - 0.0055, kmins[-1] + 0.0055)
    axes[0, 0].set_ylim(kmaxs[0] - 0.0105, kmaxs[-1] + 0.011)
    axes[0, 1].set_xlim(kmins[0] - 0.0055, kmins[-1] + 0.0055)
    axes[0, 1].set_ylim(kmaxs[0] - 0.0105, kmaxs[-1] + 0.011)

    plt.show()


def Figure15():

    stats = pd.read_csv("./Figure15.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    smins, smaxs = np.unique(stats[0]), np.unique(stats[1])

    dsmin = smins[1] - smins[0]
    dsmax = smaxs[1] - smaxs[0]

    bestmean = np.where((stats[0] == 50.0) & (stats[1] == 150.0))[0][0]

    # Upper panel
    fig, axes = plt.subplots(figsize=(7.5, 2.5), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.10, top=0.97, bottom=0.18, right=0.8, hspace=0.0, wspace=0.10)

    axes[0, 0].imshow(
        stats[2].reshape(len(smins), len(smaxs)).T,
        extent=(smins[0] - 2.0, smins[-1] + 2.0, smaxs[0] - 2.0, smaxs[-1] + 2.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.20,
        vmax=0.20,
    )
    cax = axes[0, 1].imshow(
        stats[3].reshape(len(smins), len(smaxs)).T,
        extent=(smins[0] - 2.0, smins[-1] + 2.0, smaxs[0] - 2.0, smaxs[-1] + 2.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-0.20,
        vmax=0.20,
    )
    lines = contour_rect(stats[2].reshape(len(smins), len(smaxs)).T, 0.1)
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dsmin + smins[0], np.array(line[0]) * dsmax + smaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(stats[3].reshape(len(smins), len(smaxs)).T, 0.2)
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dsmin + smins[0], np.array(line[0]) * dsmax + smaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    axes[0, 1].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    fig.supxlabel(r"$s_{\mathrm{min}}\,(h^{-1}\,\mathrm{Mpc})$", x=0.35)
    fig.supylabel(r"$s_{\mathrm{max}}\,(h^{-1}\,\mathrm{Mpc})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=r"$\Delta \alpha_{\mathrm{iso},\mathrm{ap}}\,(\%)$",
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(smins[0] - 2.4, smins[-1] + 2.4)
    axes[0, 0].set_ylim(smaxs[0] - 2.2, smaxs[-1] + 2.4)
    axes[0, 1].set_xlim(smins[0] - 2.4, smins[-1] + 2.2)
    axes[0, 1].set_ylim(smaxs[0] - 2.4, smaxs[-1] + 2.4)

    # Lower panel
    fig, axes = plt.subplots(figsize=(7.5, 2.5), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(left=0.10, top=0.97, bottom=0.18, right=0.8, hspace=0.0, wspace=0.10)

    stats[4:] /= stats[4:, bestmean][:, None]

    axes[0, 0].imshow(
        stats[4].reshape(len(smins), len(smaxs)).T,
        extent=(smins[0] - 2.0, smins[-1] + 2.0, smaxs[0] - 2.0, smaxs[-1] + 2.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.85,
        vmax=1.15,
    )
    cax = axes[0, 1].imshow(
        stats[5].reshape(len(smins), len(smaxs)).T,
        extent=(smins[0] - 2.0, smins[-1] + 2.0, smaxs[0] - 2.0, smaxs[-1] + 2.0),
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0.85,
        vmax=1.15,
    )
    lines = contour_rect(stats[2].reshape(len(smins), len(smaxs)).T, 0.1)
    for line in lines:
        axes[0, 0].plot(np.array(line[1]) * dsmin + smins[0], np.array(line[0]) * dsmax + smaxs[0], color="k", alpha=0.5, ls="--")
    lines = contour_rect(stats[3].reshape(len(smins), len(smaxs)).T, 0.2)
    for line in lines:
        axes[0, 1].plot(np.array(line[1]) * dsmin + smins[0], np.array(line[0]) * dsmax + smaxs[0], color="k", alpha=0.5, ls="--")
    axes[0, 0].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    axes[0, 1].errorbar(stats[0, bestmean], stats[1, bestmean], marker="x", color="g", markersize=8, ls="None")
    fig.supxlabel(r"$s_{\mathrm{min}}\,(h^{-1}\,\mathrm{Mpc})$", x=0.35)
    fig.supylabel(r"$s_{\mathrm{max}}\,(h^{-1}\,\mathrm{Mpc})$", y=0.55)
    fig.colorbar(
        cax,
        ax=axes.ravel().tolist(),
        label=(r"$\mathrm{Relative}\,\,\sigma_{\alpha_{\mathrm{iso},\mathrm{ap}}}$"),
    )
    axes[0, 0].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{iso}}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 1].text(
        0.95,
        0.95,
        r"$\alpha_{\mathrm{ap}}$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        color="k",
    )
    axes[0, 0].set_xlim(smins[0] - 2.4, smins[-1] + 2.4)
    axes[0, 0].set_ylim(smaxs[0] - 2.2, smaxs[-1] + 2.4)
    axes[0, 1].set_xlim(smins[0] - 2.4, smins[-1] + 2.2)
    axes[0, 1].set_ylim(smaxs[0] - 2.4, smaxs[-1] + 2.4)

    plt.show()


def Figure16():

    pk_w = pd.read_csv("./Figure16_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    xi_w = pd.read_csv("./Figure16_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T

    labels = [
        r"$\mathrm{Eisenstein\,&\,Hu\,1998}$",
        r"$\mathrm{Hinton\,et.\,al.\,2017}$",
        r"$\mathrm{Wallisch\,et.\,al.\,2018}$",
        r"$\mathrm{Brieden\,et.\,al.\,2022}$",
        r"$\mathrm{Savitsky-Golay\,Filter}$",
    ]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i, label in enumerate(labels):
        ax1.plot(pk_w[0], pk_w[0] * pk_w[i + 1], "-", label=label)
    ax1.set_xlim(0.0, 0.4)
    ax1.set_xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=14)
    ax1.set_ylabel(r"$k\,P_{\mathrm{w}}(k)\quad[h^{-2}\,\mathrm{Mpc}^{2}]$", fontsize=14)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for i, label in enumerate(labels):
        ax1.plot(xi_w[0], xi_w[0] ** 2 * xi_w[i + 1], "-", label=label)
    ax1.set_xlim(30.0, 180.0)
    ax1.set_xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=14)
    ax1.set_ylabel(r"$s^{2}\xi_{\mathrm{w}}(s)\quad[h^{-2}Mpc^{2}]$", fontsize=14)
    ax1.legend(fontsize=12)

    plt.show()


def Figure17():

    diff_pk = pd.read_csv("./Figure17_pk.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    diff_xi = pd.read_csv("./Figure17_xi.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T

    boxprops_pk = {"lw": 1.3, "color": "#1f77b4"}
    boxprops_xi = {"lw": 1.3, "color": "#ff7f0e"}
    medianprops = {"lw": 1.5, "color": "g"}
    whiskerprops = {"lw": 1.3, "color": "k"}

    bplist = []
    fig, axes = plt.subplots(figsize=(8, 10), nrows=4, ncols=1, sharex=True, squeeze=False)
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.08, right=0.95, hspace=0.0, wspace=0.0)
    plt.suptitle(r"$\mathrm{Template\,Dewiggling\,Differences}$", fontsize=16)
    for panel in range(2):
        axes[panel, 0].boxplot(
            diff_pk[panel : -2 + panel : 2].T,
            positions=np.arange(4) - 0.2,
            widths=0.2,
            whis=(0, 100),
            showfliers=False,
            boxprops=boxprops_pk,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=whiskerprops,
        )
        axes[panel, 0].boxplot(
            diff_xi[panel : -2 + panel : 2].T,
            positions=np.arange(4) + 0.2,
            widths=0.2,
            whis=(0, 100),
            showfliers=False,
            boxprops=boxprops_xi,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=whiskerprops,
        )
        bp1 = axes[panel + 2, 0].boxplot(
            (diff_pk[panel : -2 + panel : 2] / diff_pk[-2 + panel]).T,
            positions=np.arange(4) - 0.2,
            widths=0.2,
            whis=(0, 100),
            showfliers=False,
            boxprops=boxprops_pk,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=whiskerprops,
        )
        bp2 = axes[panel + 2, 0].boxplot(
            (diff_xi[panel : -2 + panel : 2] / diff_xi[-2 + panel]).T,
            positions=np.arange(4) + 0.2,
            widths=0.2,
            whis=(0, 100),
            showfliers=False,
            boxprops=boxprops_xi,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=whiskerprops,
        )
        if panel == 1:
            bplist.append(bp1["boxes"][0])
            bplist.append(bp2["boxes"][0])
    for panel in [0, 2, 3]:
        axes[panel, 0].axhline(y=0.0, color="k", ls="-", zorder=0, lw=1.5, alpha=0.75)
        axes[panel, 0].axhline(y=-0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
        axes[panel, 0].axhline(y=0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=0.0, color="k", ls="-", zorder=0, lw=1.5, alpha=0.75)
    axes[1, 0].axhline(y=-0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)

    axes[0, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{iso}} (\%)$", fontsize=16)
    axes[1, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{ap}} (\%)$", fontsize=16)
    axes[2, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{iso}} / \sigma_{\alpha_{\mathrm{iso}}}$", fontsize=16)
    axes[3, 0].set_ylabel(r"$\Delta\alpha_{\mathrm{ap}} / \sigma_{\alpha_{\mathrm{ap}}}$", fontsize=16)
    axes[3, 0].set_xlabel(r"$\mathrm{Method}$", fontsize=16)
    axes[3, 0].legend(bplist, [r"$P(k)$", r"$\xi(s)$"], ncol=2, fontsize=12)
    plt.setp(
        axes,
        xticks=np.arange(4),
        xticklabels=[
            r"$\mathrm{Eisenstein\,&\,Hu,}\,1998$",
            r"$\mathrm{Hinton\,et.\,al.,}\,2017$",
            r"$\mathrm{Brieden\,et.\,al.,}\,2022$",
            r"$\mathrm{Savitsky-Golay}$",
        ],
    )
    plt.setp(axes[3, 0].get_xticklabels(), rotation=30, fontsize=12)
    for panel in range(4):
        axes[panel, 0].xaxis.set_tick_params(labelsize=12)
        axes[panel, 0].yaxis.set_tick_params(labelsize=12)
        for axis in ["top", "bottom", "left", "right"]:
            axes[panel, 0].spines[axis].set_linewidth(1.3)

    fig.savefig("./Figure17.png", bbox_inches="tight", transparent=True, dpi=300)


def Figure18():

    stats_zeus = pd.read_csv("./Figure18_zeus.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    stats_emcee = pd.read_csv("./Figure18_emcee.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    stats_dynesty_static = pd.read_csv("./Figure18_dynesty_static.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T
    stats_dynesty_dynamic = pd.read_csv("./Figure18_dynesty_dynamic.txt", sep=r"\s+", header=None, skiprows=1).to_numpy().T

    sampler_names = [
        r"$\mathrm{Zeus}$",
        r"$\mathrm{Emcee}$",
        r"$\mathrm{Dynesty\,Static}$",
        r"$\mathrm{Dynesty\,Dynamic}$",
    ]

    boxprops_pk = {"lw": 1.3, "color": "#1f77b4"}
    boxprops_xi = {"lw": 1.3, "color": "#ff7f0e"}
    medianprops = {"lw": 1.5, "color": "g"}
    whiskerprops = {"lw": 1.3, "color": "k"}

    bplist = []
    fig, axes = plt.subplots(figsize=(6, 10), nrows=6, ncols=1, sharex=True, squeeze=False)
    plt.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95, hspace=0.0, wspace=0.0)
    for panel in range(6):
        axes[panel, 0].axhline(0.0, color="k", ls="-", zorder=0, lw=0.75)
        for i, stats in enumerate([stats_zeus, stats_emcee, stats_dynesty_static, stats_dynesty_dynamic]):
            bp1 = axes[panel, 0].boxplot(
                stats[panel].T,
                positions=[i - 0.2],
                widths=0.2,
                whis=(0, 100),
                showfliers=False,
                boxprops=boxprops_pk,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                capprops=whiskerprops,
            )
            bp2 = axes[panel, 0].boxplot(
                stats[panel + 6].T,
                positions=[i + 0.2],
                widths=0.2,
                whis=(0, 100),
                showfliers=False,
                boxprops=boxprops_xi,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                capprops=whiskerprops,
            )
            if panel == 5 and i == 0:
                bplist.append(bp1["boxes"][0])
                bplist.append(bp2["boxes"][0])

    axes[0, 0].set_ylabel(r"$\Delta \alpha_{\mathrm{iso}} (\%)$", fontsize=16)
    axes[1, 0].set_ylabel(r"$\Delta \alpha_{\mathrm{ap}} (\%)$", fontsize=16)
    axes[2, 0].set_ylabel(r"$\Delta \sigma^{68\%}_{\alpha_{\mathrm{iso}}} (\%)$", fontsize=16)
    axes[3, 0].set_ylabel(r"$\Delta \sigma^{68\%}_{\alpha_{\mathrm{ap}}} (\%)$", fontsize=16)
    axes[4, 0].set_ylabel(r"$\Delta \sigma^{95\%}_{\alpha_{\mathrm{iso}}} (\%)$", fontsize=16)
    axes[5, 0].set_ylabel(r"$\Delta \sigma^{95\%}_{\alpha_{\mathrm{ap}}} (\%)$", fontsize=16)
    axes[5, 0].set_xlabel(r"$\mathrm{Sampler}$", fontsize=16)
    axes[0, 0].axhline(y=-0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[0, 0].axhline(y=0.1, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=-0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[1, 0].axhline(y=0.2, color="k", ls="--", zorder=0, lw=1.2, alpha=0.75)
    axes[3, 0].legend(
        bplist,
        [r"$P(k)$", r"$\xi(s)$"],
        loc="center right",
        bbox_to_anchor=(1.25, 1.0),
        frameon=False,
        fontsize=14,
    )
    plt.setp(axes, xticks=[0, 1, 2, 3], xticklabels=sampler_names)
    plt.xticks(rotation=30)
    fig.savefig("./Figure18.png", bbox_inches="tight", transparent=True, dpi=300)


if __name__ == "__main__":

    # Figure7()
    # for row in ["top", "middle", "bottom"]:
    #    Figure11(row=row)
    # Figure12()
    Figure13()
    # Figure14()
    # Figure15()
    # Figure16()
    Figure17()
    Figure18()
