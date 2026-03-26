import pandas as pd
import numpy as np
from skbio.stats.distance import DistanceMatrix, permanova
from sklearn.preprocessing import StandardScaler

countries = ["China", "USA", "Philippines", "Singapore", "SouthKorea"]
groups = ["recombinant", "nonrecombinant", "mixed"] # change depending on file names

def load_country_data(country):
    dfs = []
    
    for group in groups:
        file_path = f"{country}_{group}.csv"
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            if not {"birth", "death"}.issubset(df.columns):
                continue
            
            df["group"] = group
            dfs.append(df)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(dfs) == 0:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def run_permanova_1d(df, feature_col):
    df = df[[feature_col, "group"]].dropna()
    
    values = df[feature_col].values.reshape(-1, 1)
    
    values = StandardScaler().fit_transform(values)
    
    dist_matrix = np.abs(values - values.T)
    
    ids = [f"sample_{i}" for i in range(len(values))]
    dm = DistanceMatrix(dist_matrix, ids)
    
    return permanova(dm, df["group"].values)

for country in countries:
    print(f"\n==================== {country} ====================")
    
    df = load_country_data(country)
    
    if df is None:
        print("No data loaded.")
        continue
    
    print("\n--- PERMANOVA (Birth) ---")
    print(run_permanova_1d(df, "birth"))
    
    print("\n--- PERMANOVA (Death) ---")
    print(run_permanova_1d(df, "death"))
