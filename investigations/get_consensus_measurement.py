import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import linalg

filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/xi_individual/xi_individual_alphameans.csv"

df_pk = pd.read_csv(filename_pk)
df_xi = pd.read_csv(filename_xi)

df_all = pd.merge(df_pk, df_xi, on="realisation")
use_columns = [a for a in list(df_all.columns) if "Noda" not in a and "realisation" not in a and "Recon" in a and "xi" in a]

df = df_all[use_columns]

cols_all = [a for a in df.columns if "Beutler 2017 R" not in a and "Beutler 2017 P" not in a]
cols_mean = [a for a in cols_all if "mean" in a]
cols_std = [a for a in cols_all if "std" in a]
labels = [a for a in cols_mean]
swaps = [("_pk", " $P(k)$"), ("_mean", ""), ("_xi", " $\\xi(s)$"), ("Recon", ""), ("Beutler 2017", "B17"), ("Ding 2018", "D18"), ("Seo 2016", "S16")]
for a, b in swaps:
    labels = [l.replace(a, b) for l in labels]
means = df[cols_mean].values
stds = df[cols_std].values

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)
cov = np.cov(means.T)
dcov = np.diag(np.diag(cov))
corr = np.corrcoef(means.T)


# cov[0, 0] *= 1.03
# cov[1, 1] *= 1.01
# cov[2, 2] *= 1.01
# cov[3, 3] *= 1.01
# cov[4, 4] *= 1.01
# cov[5, 5] *= 1.01
# print(cols_mean)
# print(cov)

# scale = np.sqrt(np.diag(cov)) / std
# scale[scale > 1] = 1
# print("Scale ", scale)
# scale2d = scale[:, None] @ scale[None, :]
# scale2d = np.diag(scale) ** 2
# scale2d[scale2d == 0] = 1

# cov = cov / scale2d

var = np.diag(cov)
ppp = np.zeros(cov.shape)
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        cov_ij = cov[i, j]
        minv = min(var[i], var[j])
        ppp[i, j] = cov_ij > minv

# print(ppp)
# plt.imshow(ppp)

# fig, ax = plt.subplots(figsize=(7, 7))
# sb.heatmap(pd.DataFrame(corr, columns=labels, index=labels), annot=True, fmt="0.3f", square=True, ax=ax, cbar=False)
# ax.set_ylim(len(labels) + 0.5, -0.5)
# fig.savefig("consensus_correlation.png", transparent=True, dpi=150, bbox_inches="tight")
# fig.savefig("consensus_correlation.pdf", transparent=True, dpi=150, bbox_inches="tight")

# Compute the consensus value using the equation of Winkler1981, Sanchez2016

cov_inv = linalg.inv(cov)
sigma_c = np.sum(cov_inv)
combined = np.sum(cov_inv * mean) / sigma_c
combined_err = 1.0 / np.sqrt(sigma_c)
print("Mean measurements: ", mean)
print("Consensus: ", combined)
print("Mean scatter: ", np.sqrt(np.diag(cov)))
print("Consensus err: ", combined_err)
print("ratios: ", 1.0 / np.sqrt(sigma_c * np.diag(cov)))
print("-----------------------------------------")
if True:
    from scipy.stats import multivariate_normal as mv
    from scipy.special import loggamma

    import emcee

    num_mocks = means.shape[0]
    num_params = 1
    c_p = loggamma(num_mocks / 2).real - (num_params / 2) * np.log(np.pi * (num_mocks - 1)) - loggamma((num_mocks - num_params) * 0.5).real

    def log_prob(alpha):
        diff = mean - alpha
        chi2 = diff[None, :] @ cov_inv @ diff[:, None]
        log_likelihood = c_p - (num_mocks / 2) * np.log(1 + chi2 / (num_mocks - 1))
        return log_likelihood

    nwalkers = 50
    ndim = 1

    p0 = np.random.normal(size=(nwalkers, ndim), scale=0.1, loc=1.0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain(flat=True)
    print(samples.mean(), np.std(samples))
    plt.hist(samples[:, 0], 100, color="k", histtype="step")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$p(\theta_1)$")
    plt.gca().set_yticks([])
# This makes no sense - how is the consensus measurement more biased than anything going into it??
# Mean measurements:  [1.00039962 0.99850631 0.99840441 1.00052836 1.00052664 1.00028977]
# Consensus:  1.0025964898749093
# Mean scatter:  [0.01222109 0.01262812 0.01268712 0.0133334  0.01333228 0.01336515]
# Consensus err:  0.012027931643257472
# ratios:  [0.98419472 0.952472   0.94804269 0.90209063 0.90216599 0.89994762]
