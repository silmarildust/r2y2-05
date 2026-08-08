import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO

# -----------------------------
# STEP 1 — Load sequences (Fixed for ValueError)
# -----------------------------
def load_sequences(filepath):
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        sequences.append(str(record.seq).upper())
    
    # Normalizing length ensures we don't get the "inhomogeneous shape" error
    if sequences:
        min_len = min(len(s) for s in sequences)
        sequences = [s[:min_len] for s in sequences]
    return sequences

# -----------------------------
# STEP 2 — Remove bad columns
# -----------------------------
def clean_alignment(sequences):
    if not sequences: return np.array([])
    arr = np.array([list(seq) for seq in sequences])
    weird = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N', '-', '?'}
    
    valid_indices = []
    for j in range(arr.shape[1]):
        if not (set(arr[:, j]) & weird):
            valid_indices.append(j)
    return arr[:, valid_indices]

# -----------------------------
# STEP 3 — Convert to 0/1 matrix
# -----------------------------
def to_binary_matrix(alignment):
    if alignment.size == 0: return np.array([])
    binary = []
    for col in alignment.T:
        values, counts = np.unique(col, return_counts=True)
        if len(values) > 1:
            major = values[np.argmax(counts)]
            binary.append((col != major).astype(int))
    return np.array(binary).T

# -----------------------------
# STEP 4 — Hamming distance (Scaled to the Tens Place)
# -----------------------------
def hamming_matrix(binary_matrix):
    # Setting scale to 100 keeps distances in the "tens" (0-100 scale)
    SCALE_FACTOR = 100 
    
    # pdist gives proportion; multiplying by 100 makes it a percentage
    dist_vector = pdist(binary_matrix, metric='hamming') * SCALE_FACTOR
    
    return squareform(np.round(dist_vector)).astype(int)

# -----------------------------
# STEP 5 — Persistent homology (Enabled Dim 2)
# -----------------------------
def compute_ph(distance_matrix, maxdim=2):
    # We set maxdim=2 here to search for the 2nd dimension
    # Note: Dim 2 is much slower and often empty for DNA data
    return ripser(distance_matrix, distance_matrix=True, maxdim=maxdim)['dgms']

# -----------------------------
# STEP 6 — Save barcodes
# -----------------------------
def save_barcodes(diagrams, group_name, outname):
    rows = []
    for dim in range(len(diagrams)):
        for birth, death in diagrams[dim]:
            if not np.isfinite(death): continue
            rows.append({
                "group": group_name,
                "dimension": dim,
                "birth": int(round(birth)),
                "death": int(round(death)),
                "length": int(round(death - birth))
            })
    pd.DataFrame(rows).to_csv(outname, index=False)

# -----------------------------
# KEEPING THE ORIGINAL STRUCTURE
# -----------------------------
def goal2_pipeline(filepath, group_name, out_csv):
    print(f"\nProcessing {group_name}...")

    seqs = load_sequences(filepath)
    cleaned = clean_alignment(seqs)
    binary = to_binary_matrix(cleaned)
    hd = hamming_matrix(binary)
    diagrams = compute_ph(hd)
    save_barcodes(diagrams, group_name, out_csv)

# -----------------------------
# Execution
# -----------------------------
# Use this for your different files as needed
goal2_pipeline(
    "SY2627 RUN/combined fasta input/xbc.1, ba.2, b.1.617.2_usa.FASTA", 
    "recombinant", 
    "xbc.1, ba.2, b.1.617.2-usa_recombinant.csv")

goal2_pipeline(
    "SY2627 RUN/combined fasta input/xbc.1, ba.2, b.1.617.2_usa.FASTA", 
    "nonrecombinant", 
    "xbc.1, ba.2, b.1.617.2-usa_nonrecombinant.csv")

goal2_pipeline(
    "SY2627 RUN/combined fasta input/xbc.1, ba.2, b.1.617.2_usa.FASTA",
    "mixed", 
    "xbc.1, ba.2, b.1.617.2-usa_mixed.csv")