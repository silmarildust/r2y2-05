import pandas as pd
import numpy as np
import itertools
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("country_pairs_xbc.1 (2).csv")

df.columns = df.columns.str.strip().str.lower()

COL_GROUP   = "group"
COL_BIRTH   = "birth"
COL_DEATH   = "death"
COL_DIM     = "dimension"

df[COL_GROUP] = df[COL_GROUP].astype(str).str.strip().str.lower()
df = df.dropna(subset=[COL_GROUP, COL_BIRTH, COL_DEATH, COL_DIM])

df = df[np.isfinite(df[COL_BIRTH]) & np.isfinite(df[COL_DEATH])]

df[COL_BIRTH] = df[COL_BIRTH] * 100
df[COL_DEATH] = df[COL_DEATH] * 100

print("Dimensions present:", df[COL_DIM].unique())

alpha = 0.05

for dim, df_dim in df.groupby(COL_DIM):
    print(f"\n\n==============================")
    print(f"=== ANALYSIS FOR H{int(dim)} ===")
    print(f"==============================")

    print("Total rows:", len(df_dim))
    print("Groups:", df_dim[COL_GROUP].unique())


print("Group counts:")
print(cdf[COL_GROUP].value_counts())

grouped = {g: d for g, d in df.groupby(COL_GROUP)}

if len(grouped) < 2:
    print("Not enough groups for comparison.")
    continue

for var in [COL_BIRTH, COL_DEATH]:
    print(f"\n{var.upper()}:")

    data = [g[var].values for g in grouped.values()]

    stat, p = kruskal(*data)
    print(f"Kruskal p = {p:.5f}")

    if p < alpha:
        print("Significant → pairwise tests:")

        pairs = list(itertools.combinations(grouped.keys(), 2))
        p_vals = []
        pair_names = []

        for g1, g2 in pairs:
            x = grouped[g1][var]
            y = grouped[g2][var]

            stat_u, p_u = mannwhitneyu(x, y, alternative="two-sided")

            p_vals.append(p_u)
            pair_names.append((g1, g2))

        reject, p_adj, _, _ = multipletests(
            p_vals, alpha=alpha, method="bonferroni"
        )

        for i, (g1, g2) in enumerate(pair_names):
            print(f"{g1} vs {g2}: "
                  f"raw p={p_vals[i]:.5f}, "
                  f"adj p={p_adj[i]:.5f} → "
                  f"{'SIGNIFICANT' if reject[i] else 'ns'}")

    else:
        print("Not significant")
