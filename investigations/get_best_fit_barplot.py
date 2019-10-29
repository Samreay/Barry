import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/xi_individual/xi_individual_alphameans.csv"

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

fig, axes = plt.subplots(ncols=4, figsize=(10, 2.5), sharey=True, gridspec_kw={"hspace": 0, "wspace": 0})

i = 0
cs = ["#262232", "#262232", "#116A71", "#48AB75", "#D1E05B"]
cs2 = ["#262232", "#262232", "#116A71", "#48AB75", "#a7b536"]

for f, x in zip([filename_pk, filename_xi], ["$P(k)$", "$\\xi(s)$"]):
    for r in ("Recon", "Prerecon"):
        df = pd.read_csv(f)
        df = df[[c for c in df.columns if "_std" in c and r in c]]

        df = df.rename(columns={c: c.split(r)[0].strip() for c in df.columns})
        print(df.head())
        mins = df.min(axis=1)

        counts = [(np.isclose(df[c], mins, atol=0.0005)).sum() for c in df.columns]
        y_pos = -np.arange(len(df.columns))

        barlist = axes[i].barh(y_pos, counts, align="center", height=0.8)
        for b, c in zip(barlist, cs):
            b.set_color(c)
        axes[i].set_title(x + " " + r)
        if i == 0:
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(df.columns, fontsize=14)
            for l, c in zip(axes[i].get_yticklabels(), cs2):
                l.set_color(c)
        i += 1

fig.savefig("best_methods.png", bbox_inches="tight", transparent=True, dpi=300)
fig.savefig("best_methods.pdf", bbox_inches="tight", transparent=True, dpi=300)
plt.show()

# recon_cols = [c for c in df.columns if "Recon" in c]
#
# print(models)
# print(df[[f"{model}Recon_xi_mean" for model in models]].mean())
# print(df[[f"{model}Recon_xi_std" for model in models]].mean())
# consensus = []
# for index, row in df.iterrows():
#     evidence = [row[f"{model}Recon_xi_evidence"] for model in models]
#     total_weight = logsumexp(evidence)
#     weights = np.exp(np.array(evidence) - total_weight)
#     c = np.sum(weights * np.array([row[f"{model}Recon_xi_mean"] for model in models]))
#     s = np.sum(weights * np.array([row[f"{model}Recon_xi_std"] for model in models]))
#     consensus.append([c, s])
# print(np.mean(consensus, axis=0))
