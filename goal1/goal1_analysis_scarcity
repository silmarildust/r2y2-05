!pip install scikit-bio

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from skbio.stats.distance import DistanceMatrix, permanova

np.random.seed(42)

# Load data
df = pd.read_csv("scarcity_levels.csv")

def get_points(subdf):
    return subdf[['birth', 'death']].to_numpy()

results = {}

for dim in df['dimension'].unique():
    print(f"\n==============================")
    print(f"Dimension {dim} ANALYSIS")
    print(f"==============================")

    dim_df = df[df['dimension'] == dim].reset_index(drop=True)

    noise_levels = dim_df['level_value'].unique()

    diagrams = []
    group_labels = []
    replicate_ids = []

    # Collect diagrams
    for level in noise_levels:
        level_df = dim_df[dim_df['level_value'] == level]
        for rep in level_df['replicate'].unique():
            rep_df = level_df[level_df['replicate'] == rep]
            diagrams.append(get_points(rep_df).flatten())
            group_labels.append(str(level))
            replicate_ids.append(f"{level}_rep{rep}")

    n_samples = len(diagrams)

    # Skip if fewer than 2 total samples
    if n_samples < 2:
        print("Skipping: fewer than 2 total samples.")
        continue

    # Compute pairwise Wasserstein distance matrix
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            D[i, j] = wasserstein_distance(diagrams[i], diagrams[j])
            D[j, i] = D[i, j]

    dm = DistanceMatrix(D, ids=replicate_ids)

    grouping = pd.Series(group_labels, index=replicate_ids, name='scarcity_level')

    # Keep only groups with ≥2 replicates
    group_counts = grouping.value_counts()
    valid_groups = group_counts[group_counts >= 2].index

    if len(valid_groups) < 2:
        print("Skipping: need at least two groups with ≥2 replicates.")
        continue

    valid_ids = [rid for rid in replicate_ids if grouping[rid] in valid_groups]
    indices = [replicate_ids.index(rid) for rid in valid_ids]

    dm_filtered = DistanceMatrix(D[np.ix_(indices, indices)], ids=valid_ids)
    grouping_filtered = grouping[valid_ids]

    N = len(grouping_filtered)
    g = grouping_filtered.nunique()

    # Must have within-group degrees of freedom
    if N <= g:
        print("Skipping: N <= number of groups (no within-group variance possible).")
        continue

    # Skip if all distances are zero
    if np.all(dm_filtered.data == 0):
        print("Skipping: all pairwise distances are zero.")
        continue

    # Run PERMANOVA
    try:
        res = permanova(
            distance_matrix=dm_filtered,
            grouping=grouping_filtered,
            permutations=999
        )
    except Exception as e:
        print(f"Skipping due to PERMANOVA error: {e}")
        continue

    F_stat = res['test statistic']
    p_val = res['p-value']

    if np.isnan(F_stat):
        print("Skipping: F-statistic undefined (zero within-group variance).")
        continue

    # Compute R²
    df_between = g - 1
    df_within = N - g
    R2 = (F_stat * df_between) / (F_stat * df_between + df_within)

    # Store results
    results[dim] = {
        "F": F_stat,
        "p": p_val,
        "R2": R2
    }

    # Print results
    print(f"Samples:      {N}")
    print(f"Groups:       {g}")
    print(f"F-statistic:  {F_stat:.4f}")
    print(f"p-value:      {p_val:.6f}")
    print(f"R^2:          {R2:.4f}")

    if p_val < 0.05:
        print("Reject H0: topology is not robust to data scarcity")
    else:
        print("Fail to reject H0: topology is robust to data scarcity")
