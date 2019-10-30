import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg, stats
from scipy.optimize import minimize

filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/xi_individual/xi_individual_alphameans.csv"

df_pk = pd.read_csv(filename_pk)
df_xi = pd.read_csv(filename_xi)

df_all = pd.merge(df_pk, df_xi, on="realisation")
print(list(df_all.columns))
use_columns = [a for a in list(df_all.columns) if "realisation" not in a and "Recon" in a]

df = df_all[use_columns]

cols_mean = [a for a in df.columns if "mean" in a]
cols_std = [a for a in df.columns if "std" in a]
cols_mean_pk = [a for a in df.columns if "mean" in a and "pk" in a]
cols_std_pk = [a for a in df.columns if "std" in a and "pk" in a]
cols_mean_xi = [a for a in df.columns if "mean" in a and "xi" in a]
cols_std_xi = [a for a in df.columns if "std" in a and "xi" in a]
labels = [a for a in cols_mean]
swaps = [("_pk", " $P(k)$"), ("_mean", ""), ("_xi", " $\\xi(s)$"), ("Recon", ""), ("Beutler 2017", "B17"), ("Ding 2018", "D18"), ("Seo 2016", "S16")]
for a, b in swaps:
    labels = [l.replace(a, b) for l in labels]
means = [df[cols_mean].values, df[cols_mean_pk].values, df[cols_mean_xi].values]
stds = [df[cols_std].values, df[cols_std_pk].values, df[cols_std_xi].values]
mean = [np.mean(means[0], axis=0), np.mean(means[1], axis=0), np.mean(means[2], axis=0)]
cov = [np.cov(means[0].T), np.cov(means[1].T), np.cov(means[2].T)]
dcov = [np.diag(np.diag(cov[0])), np.diag(np.diag(cov[1])), np.diag(np.diag(cov[2]))]
corr = [np.corrcoef(means[0].T), np.corrcoef(means[1].T), np.corrcoef(means[2].T)]

# Check the ks-test p-value for each model against a chi_squared distribution and make a plot of the chi-squared values
chi2 = np.empty(np.shape(means[0]))
pvals = np.empty(len(means[0][0, 0:]))
rv = stats.chi2(1)
for i in range(len(cols_mean)):
    chi2[0:, i] = (means[0][0:, i] - 1.0) ** 2 / stds[0][0:, i] ** 2
    anderson = stats.anderson_ksamp([chi2[0:, i], rv.rvs(len(chi2[0:, i]))])
    pvals[i] = stats.kstest(chi2[0:, i], rv.cdf).pvalue
    print(cols_mean[i], np.mean(means[0][0:, i]), np.std(means[0][0:, i]), np.mean(stds[0][0:, i]), pvals[i], anderson[0], anderson[1])

if True:

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    nrows = np.amax([len(cols_mean_pk), len(cols_mean_xi)])

    fig, axes = plt.subplots(figsize=(5, 7), nrows=1, sharex=True, sharey=True)
    inner = gridspec.GridSpecFromSubplotSpec(nrows, 2, subplot_spec=axes, hspace=0.0, wspace=0.0)
    ax = plt.subplot(inner[0:])
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.set_ylabel(r"$\mathrm{CDF}(\chi^{2})$", fontsize=8)
    ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

    counter = 0
    colors = ["#262232", "#262232", "#116A71", "#48AB75", "#D1E05B"]

    y = np.linspace(0.0, 1.0, len(means[0]))
    for i in range(2):
        for j in range(nrows):
            ax = fig.add_subplot(inner[j, i])

            if i == 1 and j == nrows - 1:
                ax.spines["top"].set_color("none")
                ax.spines["bottom"].set_color("none")
                ax.spines["left"].set_color("none")
                ax.spines["right"].set_color("none")
                ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax.tick_params(axis="y", which="both", labelcolor="none", bottom=False, labelbottom=False)
                ax.set_xticks([])
                for k in range(nrows):
                    ax.plot([], label=cols_mean[k], color=colors[k])
                ax.legend(frameon=False, markerfirst=False, loc="lower right", fontsize=6)

            else:
                x = np.sort(chi2[0:, i * nrows + j])
                xvals = np.logspace(-3.0, np.log10(25.0), 10000)
                bins = np.logspace(-3.0, np.log10(25.0), 100)

                # PDF
                # ax.hist(chi2[0:, i * nrows + j], bins=bins, label=cols_mean[i * nrows + j], color=colors[j], histtype="step", density=True)
                # ax.plot(xvals, rv.pdf(xvals), color="k", linestyle="--", zorder=0, alpha=0.5, linewidth=1.5)
                # ax.set_yscale("log")

                # CDF
                ax.plot(x, y, color=colors[j], label=cols_mean[i * nrows + j], linestyle="steps", zorder=1, linewidth=1.8)
                ax.plot(xvals, rv.cdf(xvals), color="k", linestyle="--", zorder=0, alpha=0.5, linewidth=1.5)
                ax.set_ylim(0.0, 1.0)

                ax.set_xscale("log")
                ax.set_xlim(0.0008, 25.0)

                if i == 0:
                    if j == 0:
                        ax.set_title(r"$P(k)$", fontsize=8)
                    if j == nrows - 1:
                        ax.set_xlabel(r"$\chi^{2}$", fontsize=8)
                        ax.tick_params(axis="x", which="both", labelsize=8)

                    else:
                        ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)
                else:
                    if j == 0:
                        ax.set_title(r"$\xi(s)$", fontsize=8)
                    ax.tick_params(axis="y", which="both", labelcolor="none", bottom=False, labelbottom=False)
                    if j == nrows - 2:
                        ax.set_xlabel(r"$\chi^{2}$", fontsize=8)
                        ax.tick_params(axis="x", which="both", labelsize=8)
                    else:
                        ax.tick_params(axis="x", which="both", labelcolor="none", bottom=False, labelbottom=False)

    plt.savefig("alpha_individual_chi2_cdf.pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.savefig("alpha_individual_chi2_cdf.png", bbox_inches="tight", dpi=300, transparent=True)

# These results suggest simply choosing the one with the smallest error is a bad idea. This is because the Seo and Ding P(k) models
# (and the Noda model to some extent) don't seem to follow a chi^2 distribution. They are probably underestimating their uncertainties.
# So let's see if we can make them agree with the chi^2 distribution better by adding some extra uncertainty to each mock.

# For each model fit an additional error component to minimize the k-s statistic
def logchi(param):
    if 10.0 ** param[0] < 0:
        return -np.inf
    newvar = stds[0][0:, i] ** 2 + (10.0 ** param[0]) ** 2
    chi2 = (means[0][0:, i] - 1.0) ** 2 / newvar
    return stats.kstest(chi2, rv.cdf).statistic


add_err = [np.empty(len(cols_mean)), np.empty(len(cols_mean_pk)), np.empty(len(cols_mean_xi))]
for i in range(len(cols_mean)):
    add_err[0][i] = 10.0 ** (minimize(logchi, -4.0, method="Nelder-Mead", options={"xatol": 1.0e-6, "maxiter": 10000}).x)
    print(cols_mean[i], add_err[0][i])
add_err[1] = add_err[0][0 : len(cols_mean_pk)]
add_err[2] = add_err[0][len(cols_mean_pk) :]

# Turns out we can. Adding a small additional error to Seo, Ding and Noda models significantly improves the distribution
# compared to the expected chi-squared distribution.

# So how about computing consensus measurements? The above results would suggest using the BLUES method to get a consensus,
# or taking the minimum before adding some additional error to some models woulc be a bad idea. Let's test this.

# Compute consensus measurement for each mock by choosing either the measurement with the smallest error or
# combining them using the BLUES method. Do this before and after adding in the extra error determined above.
best_val = np.empty((3, len(means[0])))
best_err = np.empty((3, len(means[0])))
best_combined_val = np.empty((3, len(means[0])))
best_combined_err = np.empty((3, len(means[0])))

best_add_val = np.empty((3, len(means[0])))
best_add_err = np.empty((3, len(means[0])))
best_add_combined_val = np.empty((3, len(means[0])))
best_add_combined_err = np.empty((3, len(means[0])))
for i in range(3):
    for j, (std_old, val) in enumerate(zip(stds[i], means[i])):
        for add in [False, True]:
            if add:
                std = np.sqrt(std_old ** 2 + add_err[i] ** 2)
            else:
                std = std_old
            newcov = (std * corr[i]).T * std
            min_ind = np.argmin(std)
            cov_inv = linalg.inv(newcov)

            # Compute the BLUE coefficients for this combination and get the full combination
            weight = np.sum(cov_inv, axis=0) / np.sum(cov_inv)
            if add:
                best_add_val[i, j] = val[min_ind]
                best_add_err[i, j] = std[min_ind]
                best_add_combined_val[i, j] = np.sum(weight * val)
                best_add_combined_err[i, j] = np.sqrt(weight @ newcov @ weight)
            else:
                best_val[i, j] = val[min_ind]
                best_err[i, j] = std[min_ind]
                best_combined_val[i, j] = np.sum(weight * val)
                best_combined_err[i, j] = np.sqrt(weight @ newcov @ weight)

# An alternative way to account for missing uncertainty is to simply integrate over the different results. So try this too.
from barry.config import weighted_avg_and_std

nalpha = 100
alpha_grid = np.linspace(0.8, 1.2, nalpha)
integrate_val = np.empty((3, len(means[0])))
integrate_err = np.empty((3, len(means[0])))
for i in range(3):
    for j, (std, val) in enumerate(zip(stds[i], means[i])):
        posteriors = 1.0 / (np.sqrt(2.0 * np.pi) * std) * np.exp(-0.5 * ((val - np.tile(alpha_grid, (len(val), 1)).T) ** 2 / std ** 2))
        combined_posterior = np.sum(posteriors, axis=1) / len(std)
        norm = np.sum(combined_posterior) * (alpha_grid[-1] - alpha_grid[0]) / nalpha
        combined_posterior /= norm
        integrate_val[i, j], integrate_err[i, j] = weighted_avg_and_std(alpha_grid, combined_posterior)

# Now have a look at the distributions for each of the three methods
for i, (index, type) in enumerate(zip([1, 2, 0], ["Pk", "Xi", "All"])):
    best_chi = (best_val[index, 0:] - 1.0) ** 2 / best_err[index, 0:] ** 2
    best_combined_chi = (best_combined_val[index, 0:] - 1.0) ** 2 / best_combined_err[index, 0:] ** 2

    best_add_chi = (best_add_val[index, 0:] - 1.0) ** 2 / best_add_err[index, 0:] ** 2
    best_add_combined_chi = (best_add_combined_val[index, 0:] - 1.0) ** 2 / best_add_combined_err[index, 0:] ** 2

    integrate_chi = (integrate_val[index, 0:] - 1.0) ** 2 / integrate_err[index, 0:] ** 2

    rv = stats.chi2(1)
    x = np.linspace(0.0, 10.0, 1000)
    bins = np.logspace(-3.0, np.log10(25.0), 100)

    print(type)
    print("Minimum", np.mean(best_val[index, 0:]), np.std(best_val[index, 0:]), np.mean(best_err[index, 0:]), stats.kstest(best_chi, rv.cdf))
    print(
        "BLUES",
        np.mean(best_combined_val[index, 0:]),
        np.std(best_combined_val[index, 0:]),
        np.mean(best_combined_err[index, 0:]),
        stats.kstest(best_combined_chi, rv.cdf),
    )
    print(
        "Minimum-add", np.mean(best_add_val[index, 0:]), np.std(best_add_val[index, 0:]), np.mean(best_add_err[index, 0:]), stats.kstest(best_add_chi, rv.cdf)
    )
    print(
        "BLUES-add",
        np.mean(best_add_combined_val[index, 0:]),
        np.std(best_add_combined_val[index, 0:]),
        np.mean(best_add_combined_err[index, 0:]),
        stats.kstest(best_add_combined_chi, rv.cdf),
    )

    print(
        "Integration",
        np.mean(integrate_val[index, 0:]),
        np.std(integrate_val[index, 0:]),
        np.mean(integrate_err[index, 0:]),
        stats.kstest(integrate_chi, rv.cdf),
    )
    print()

# The results show that the BLUE method doesn't recover a chi-squared distribution before or after including the additional errors on
# the models. It also doesn't work for the correlation function, even though the models were all good. This suggests some internal tension
# I did try to check for this on a mock by mock basis, but couldn't come up with a decent automated method. Even comparing each model for a given
# mock using their individual chi-squared differences didn't solve this, unless you were super (too!) strict about how close to models had to agree.

# However, good choices are to take the minimum error after including the additive component, or using the integration method.

# Overall this seems to suggest that differences in P(k) models are not telling you about additional information between methods, but
# rather that the different methods have some small systematic errors. This may not be true adding P(k) and Xi(s), these could be
# accessing slightly different information. So how about we try combining the Minimum-add version of both of these using BLUES?
pkxi_corr = np.corrcoef([best_add_val[1, 0:], best_add_val[2, 0:]])
pk_xi_best_val = np.empty(len(means[0]))
pk_xi_best_err = np.empty(len(means[0]))
pk_xi_combined_val = np.empty(len(means[0]))
pk_xi_combined_err = np.empty(len(means[0]))
for j, (val, std) in enumerate(zip(np.column_stack((best_add_val[1], best_add_val[2])), np.column_stack((best_add_err[1], best_add_err[2])))):
    newcov = (std * pkxi_corr).T * std
    min_ind = np.argmin(std)
    cov_inv = linalg.inv(newcov)

    # Compute the BLUE coefficients for this combination and get the full combination
    weight = np.sum(cov_inv, axis=0) / np.sum(cov_inv)
    pk_xi_best_val[j] = val[min_ind]
    pk_xi_best_err[j] = std[min_ind]
    pk_xi_combined_val[j] = np.sum(weight * val)
    pk_xi_combined_err[j] = np.sqrt(weight @ newcov @ weight)

pk_xi_best_chi = (pk_xi_best_val - 1.0) ** 2 / pk_xi_best_err ** 2
pk_xi_combined_chi = (pk_xi_combined_val - 1.0) ** 2 / pk_xi_combined_err ** 2
print("Pk_err < Xi_err: ", len(np.where(best_add_err[1] < best_add_err[2])[0]))
print("Pk+Xi Minimum", np.mean(pk_xi_best_val), np.std(pk_xi_best_val), np.mean(pk_xi_best_err), stats.kstest(pk_xi_best_chi, rv.cdf))
print("Pk+Xi BLUES", np.mean(pk_xi_combined_val), np.std(pk_xi_combined_val), np.mean(pk_xi_combined_err), stats.kstest(pk_xi_combined_chi, rv.cdf))

# Finally make a couple of plots comparing errors on each mock when we
# 1: Simply average of all 9 models for power spectrum /correlation function
# 2: Take the smallest error model for each mock after adding on any additional error required
# 3: Calculate the BLUES combination of P(k) and Xi after adding on the additional error required for each model.
if True:

    colors = ["#262232", "#116A71", "#48AB75", "#D1E05B"]

    fig, axes = plt.subplots(figsize=(5, 4), nrows=1, sharex=True)
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axes, hspace=0.0, wspace=0.0, width_ratios=[3, 1])
    ax = plt.subplot(inner[0:])
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

    for i in range(2):
        step = 1
        x = np.linspace(1, len(means[0]), len(means[0]))[::step]
        if i == 0:
            err_rat = pk_xi_combined_err / integrate_err[0]
        else:
            err_rat = pk_xi_combined_err / pk_xi_best_err

        ax = fig.add_subplot(inner[i, 0])
        ax.scatter(x, err_rat[::step], c=np.abs(pk_xi_combined_val - integrate_val[0]), marker="o", s=2, vmin=0.0)
        ax.axhline(1, c="k", lw=1, ls="--")
        ax.set_ylim(0.5, 1.05)
        if i == 0:
            ax.set_xticklabels([])
            ax.set_ylabel(r"$\sigma_{\alpha}^{\mathrm{BLUES}} / \sigma_{\alpha}^{\mathrm{average}}$")
        else:
            ax.set_xlabel("Realisation", fontsize=14)
            ax.set_ylabel(r"$\sigma_{\alpha}^{\mathrm{BLUES}} / \sigma_{\alpha}^{\mathrm{min(P,\xi)}}$")

    for i in range(2):
        step = 1
        x = np.linspace(1, len(means[0]), len(means[0]))[::step]
        if i == 0:
            err_rat = pk_xi_combined_err / integrate_err[0]
        else:
            err_rat = pk_xi_combined_err / pk_xi_best_err

        ax = fig.add_subplot(inner[i, 1])
        ax.hist(err_rat, bins=50, histtype="stepfilled", linewidth=2, alpha=0.3, color=colors[1], orientation="horizontal", log=True)
        ax.hist(err_rat, bins=50, histtype="step", linewidth=1.5, color=colors[1], orientation="horizontal", log=True)
        ax.axhline(1, c="k", lw=1, ls="--")
        ax.set_ylim(0.5, 1.05)
        ax.set_yticklabels([])
        if i == 0:
            ax.set_xticklabels([])

    plt.savefig("consensus_individual.pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.savefig("consensus_individual.png", bbox_inches="tight", dpi=300, transparent=True)
