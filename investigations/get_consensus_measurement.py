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

cols = [a for a in df.columns if "mean" in a]
labels = [a.replace("_pk", " $P(k)$").replace("_mean", "").replace("_xi", " $\\xi(s)$").replace("Recon", "") for a in cols]
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

# Answer: Yes, by between 5-10%
