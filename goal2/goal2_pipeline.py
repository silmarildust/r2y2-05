import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

# -----------------------------
# STEP 1 — Load sequences
# -----------------------------
def load_sequences(filepath):
    df = pd.read_csv(filepath, sep=';')
    sequences = df['seqName'].astype(str).tolist()
    return sequences


# -----------------------------
# STEP 2 — Remove bad columns
# -----------------------------
def clean_alignment(sequences):
    arr = np.array([list(seq) for seq in sequences])
    
    valid_chars = set(['A', 'T', 'C', 'G', '-'])
    
    keep_cols = []
    for col in range(arr.shape[1]):
        column = set(arr[:, col])
        if column.issubset(valid_chars):
            keep_cols.append(col)

    cleaned = arr[:, keep_cols]
    return cleaned


# -----------------------------
# STEP 3 — Convert to 0/1 matrix (biallelic)
# -----------------------------
def to_binary_matrix(alignment):
    binary = []

    for col in alignment.T:
        values, counts = np.unique(col, return_counts=True)
        major = values[np.argmax(counts)]
        bin_col = (col != major).astype(int)
        binary.append(bin_col)

    return np.array(binary).T


# -----------------------------
# STEP 4 — Hamming distance
# -----------------------------
def hamming_matrix(binary_matrix):
    return squareform(pdist(binary_matrix, metric='hamming'))


# -----------------------------
# STEP 5 — Persistent homology
# -----------------------------
def compute_ph(distance_matrix, maxdim=2):
    diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim)['dgms']
    return diagrams


# -----------------------------
# STEP 6 — Save barcodes
# -----------------------------
def save_barcodes(diagrams, group_name, outname):
    rows = []

    for dim in range(len(diagrams)):
        for birth, death in diagrams[dim]:
            if not np.isfinite(death):
                continue

            rows.append({
                "group": group_name,
                "dimension": dim,
                "birth": birth,
                "death": death,
                "length": death - birth
            })

    df = pd.DataFrame(rows)
    df.to_csv(outname, index=False)
    print(f"Saved {outname}")


# -----------------------------
# FULL PIPELINE FOR ONE GROUP
# -----------------------------
def goal2_pipeline(filepath, group_name, out_csv):
    print(f"\nProcessing {group_name}...")

    seqs = load_sequences(filepath)
    cleaned = clean_alignment(seqs)
    binary = to_binary_matrix(cleaned)
    hd = hamming_matrix(binary)
    diagrams = compute_ph(hd)
    save_barcodes(diagrams, group_name, out_csv)

goal2_pipeline(
    "goal2_data/recombinant.csv",
    "recombinant",
    "recombinant_barcodes.csv"
)

goal2_pipeline(
    "goal2_data/nonrecombinant.csv",
    "nonrecombinant",
    "nonrecombinant_barcodes.csv"
)

goal2_pipeline(
    "goal2_data/mixed.csv",
    "mixed",
    "mixed_barcodes.csv"

)
