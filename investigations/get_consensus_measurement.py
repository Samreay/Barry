import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/xi_individual/xi_individual_alphameans.csv"

df_pk = pd.read_csv(filename_pk)
df_xi = pd.read_csv(filename_xi)

df_all = pd.merge(df_pk, df_xi, on="realisation")
use_columns = [a for a in list(df_all.columns) if "Noda" not in a and "realisation" not in a and "Recon" in a]

df = df_all[use_columns]

cols = [a for a in df.columns if "mean" in a and "Beutler 2017 R" not in a and "Beutler 2017 P" not in a]
labels = [a for a in cols]
swaps = [("_pk", " $P(k)$"), ("_mean", ""), ("_xi", " $\\xi(s)$"), ("Recon", ""), ("Beutler 2017", "B17"), ("Ding 2018", "D18"), ("Seo 2016", "S16")]
for a, b in swaps:
    labels = [l.replace(a, b) for l in labels]
means = df[cols].values
stds = df[[a for a in df.columns if "std" in a]].values

mean = np.mean(means, axis=0)
cov = np.cov(means.T)
corr = np.corrcoef(means.T)

fig, ax = plt.subplots(figsize=(7, 7))
sb.heatmap(pd.DataFrame(corr, columns=labels, index=labels), annot=True, fmt="0.3f", square=True, ax=ax, cbar=False)
ax.set_ylim(len(labels) + 0.5, -0.5)
fig.savefig("consensus_correlation.png", transparent=True, dpi=150, bbox_inches="tight")
fig.savefig("consensus_correlation.pdf", transparent=True, dpi=150, bbox_inches="tight")

# Compute the consensus value using the equation of Winkler1981, Sanchez2016
from scipy import linalg

cov_inv = linalg.inv(cov)
sigma_c = np.sum(cov_inv)
combined = np.sum(cov_inv * mean) / sigma_c
combined_err = 1.0 / np.sqrt(sigma_c)
print("Mean measurements: ", mean)
print("Consensus: ", combined)
print("Mean scatter: ", np.sqrt(np.diag(cov)))
print("Consensus err: ", combined_err)
print("ratios: ", 1.0 / np.sqrt(sigma_c * np.diag(cov)))

# This makes no sense - how is the consensus measurement more biased than anything going into it??
# Mean measurements:  [1.00039962 0.99850631 0.99840441 1.00052836 1.00052664 1.00028977]
# Consensus:  1.0025964898749093
# Mean scatter:  [0.01222109 0.01262812 0.01268712 0.0133334  0.01333228 0.01336515]
# Consensus err:  0.012027931643257472
# ratios:  [0.98419472 0.952472   0.94804269 0.90209063 0.90216599 0.89994762]
