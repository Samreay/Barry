import pandas as pd

filename_pk = "../config/plots/pk_individual/pk_individual_alphameans.csv"
filename_xi = "../config/plots/pk_individual/pk_individual_alphameans.csv"

df_pk = pd.read_csv(filename_pk)
df_xi = pd.read_csv(filename_xi)

df_all = pd.merge(df_pk, df_xi, on="realisation")
use_columns = [a for a in list(df_all.columns) if "Noda" not in a and "realisation" not in a and "Recon" in a]

df = df_all[use_columns]

means = df[[a for a in df.columns if "mean" in a]]
stds = df[[a for a in df.columns if "std" in a]]
