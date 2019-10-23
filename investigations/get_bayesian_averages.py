import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/xi_individual/xi_individual_alphameans.csv"


df = pd.read_csv(filename_xi)

cols = [c for c in df.columns if "evidence" in c and "Recon" in c and "2017 R" not in c]
df2 = df[cols]
fig, ax = plt.subplots(figsize=(20, 6))
for c in df2.columns:
    ax.plot(df2[c][:100], label=c)
ax.legend()

models = list(set([c.split("Recon")[0] for c in df.columns if "Recon" in c and "2017 R" not in c]))
recon_cols = [c for c in df.columns if "Recon" in c]

print(models)
print(df[[f"{model}Recon_xi_mean" for model in models]].mean())
print(df[[f"{model}Recon_xi_std" for model in models]].mean())
consensus = []
for index, row in df.iterrows():
    evidence = [row[f"{model}Recon_xi_evidence"] for model in models]
    total_weight = logsumexp(evidence)
    weights = np.exp(np.array(evidence) - total_weight)
    c = np.sum(weights * np.array([row[f"{model}Recon_xi_mean"] for model in models]))
    s = np.sum(weights * np.array([row[f"{model}Recon_xi_std"] for model in models]))
    consensus.append([c, s])
print(np.mean(consensus, axis=0))
