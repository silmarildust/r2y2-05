import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from skbio.stats.distance import DistanceMatrix, permanova
from itertools import combinations

np.random.seed(42)

files = {
    "recombinant": "goal2_recombinant_barcodes.csv",
    "nonrecombinant": "goal2_nonrecombinant_barcodes.csv",
    "mixed": "goal2_mixed_barcodes.csv"
}

dfs = {name: pd.read_csv(path) for name, path in files.items()}

def get_points(subdf):
    """Flatten birth/death coordinates for Wasserstein."""
    return subdf[['birth', 'death']].to_numpy().flatten()

pairs = list(combinations(dfs.keys(), 2))  
comparison_results = {}

for name1, name2 in pairs:
    print()
    print(f"Pairwise comparison: {name1} vs {name2}")
    print()
    
    df1, df2 = dfs[name1], dfs[name2]
    dimensions = sorted(set(df1['dimension'].unique()) & set(df2['dimension'].unique()))
    
    pair_results = {}
    
    for dim in dimensions:
        dim_df1 = df1[df1['dimension'] == dim]
        dim_df2 = df2[df2['dimension'] == dim]
        
        diagrams = []
        group_labels = []
        replicate_ids = []
        
        # Add group 1
        for grp in dim_df1['group'].unique():
            grp_df = dim_df1[dim_df1['group'] == grp]
            diagrams.append(get_points(grp_df))
            group_labels.append(name1)
            replicate_ids.append(f"{name1}_{grp}")
        
        # Add group 2
        for grp in dim_df2['group'].unique():
            grp_df = dim_df2[dim_df2['group'] == grp]
            diagrams.append(get_points(grp_df))
            group_labels.append(name2)
            replicate_ids.append(f"{name2}_{grp}")
        
        if len(diagrams) < 2:
            pair_results[dim] = {"p_value": None, "significant": None, "note": "Not enough samples"}
            continue
        
        n_samples = len(diagrams)
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = wasserstein_distance(diagrams[i], diagrams[j])
                D[j, i] = D[i, j]
        
        dm = DistanceMatrix(D, ids=replicate_ids)
        grouping = pd.Series(group_labels, index=replicate_ids, name='group')
        
        group_counts = grouping.value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        if len(valid_groups) < 2:
            pair_results[dim] = {"p_value": None, "significant": None, "note": "Not enough replicates per group"}
            continue
        
        valid_ids = [rid for rid in replicate_ids if grouping[rid] in valid_groups]
        indices = [replicate_ids.index(rid) for rid in valid_ids]
        
        dm_filtered = DistanceMatrix(D[np.ix_(indices, indices)], ids=valid_ids)
        grouping_filtered = grouping[valid_ids]
        
        res = permanova(distance_matrix=dm_filtered, grouping=grouping_filtered, permutations=999)
        p_val = res['p-value']
        significant = p_val < 0.05
        
        pair_results[dim] = {"p_value": p_val, "significant": significant, "note": "H0 rejected" if significant else "H0 not rejected"}
        print(f"Dimension {dim}: p-value = {p_val:.6f} → {'Significant difference' if significant else 'No significant difference'}")
    
    comparison_results[(name1, name2)] = pair_results

print("\n=== Pairwise comparison summary (H0 rejected = significant difference) ===")
for pair, dims in comparison_results.items():
    print(f"\nPair: {pair[0]} vs {pair[1]}")
    for dim, result in dims.items():
        if result["p_value"] is not None:
            print(f"Dimension {dim}: p={result['p_value']:.4f}, {result['note']}")
        else:
            print(f"Dimension {dim}: {result['note']}")
