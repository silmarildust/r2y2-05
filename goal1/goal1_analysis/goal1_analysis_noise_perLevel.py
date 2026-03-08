import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from skbio.stats.distance import DistanceMatrix, permanova

np.random.seed(42)
df = pd.read_csv("noise_levels.csv")

def get_points(subdf):
    return subdf[['birth', 'death']].to_numpy()

robustness_profiles = {}

for dim in df['dimension'].unique():
    print()
    print(f"Dimension {dim} ANALYSIS")
    print()

    dim_df = df[df['dimension'] == dim].reset_index(drop=True)
    noise_levels = sorted(dim_df['level_value'].unique())
    profile = []

    for max_level in noise_levels:
        sub_df = dim_df[dim_df['level_value'] <= max_level]

        diagrams = []
        group_labels = []
        replicate_ids = []

        for level in sorted(sub_df['level_value'].unique()):
            level_df = sub_df[sub_df['level_value'] == level]
            for rep in level_df['replicate'].unique():
                rep_df = level_df[level_df['replicate'] == rep]
                diagrams.append(get_points(rep_df).flatten())
                group_labels.append(str(level))
                replicate_ids.append(f"{level}_rep{rep}")

        if len(diagrams) < 2:
            profile.append((max_level, None, "Not enough samples"))
            continue

        n_samples = len(diagrams)
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = wasserstein_distance(diagrams[i], diagrams[j])
                D[j, i] = D[i, j]

        dm = DistanceMatrix(D, ids=replicate_ids)
        grouping = pd.Series(group_labels, index=replicate_ids, name='noise_level')

        group_counts = grouping.value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        if len(valid_groups) < 2:
            profile.append((max_level, None, "Not enough replicates per group"))
            continue

        valid_ids = [rid for rid in replicate_ids if grouping[rid] in valid_groups]
        indices = [replicate_ids.index(rid) for rid in valid_ids]

        dm_filtered = DistanceMatrix(D[np.ix_(indices, indices)], ids=valid_ids)
        grouping_filtered = grouping[valid_ids]

        res = permanova(distance_matrix=dm_filtered,
                        grouping=grouping_filtered,
                        permutations=999)

        F = res['test statistic']
        k = res['number of groups']
        N = res['sample size']
        R2 = (F * (k - 1)) / (F * (k - 1) + (N - k))

        status = "Robust" if res['p-value'] >= 0.05 else "Not robust"
        profile.append((max_level, R2, status))
        print(f"Noise level {max_level}: R²={R2:.4f}, p={res['p-value']:.6f} → {status}")

        if status == "Not robust":
            print(f"Maximum noise where dimension {dim} is robust ≤ {max_level-5}")
            break

    robustness_profiles[dim] = profile

print("\nMaximum robust noise levels per dimension:")
for dim, profile in robustness_profiles.items():
    robust_levels = [level for level, R2, status in profile if status == "Robust"]
    if robust_levels:
        max_robust = max(robust_levels)
        print(f"Dimension {dim}: maximum robust noise ≤ {max_robust}")
    else:
        print(f"Dimension {dim}: no robust noise levels")
