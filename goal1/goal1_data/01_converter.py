import os
import re
import numpy as np
from collections import Counter

class BiallelicConverter:
    def __init__(self, sample_size=100):
        self.sample_size = sample_size

    def read_sequences(self, filename):
        """
        Read sequences from a text/FASTA file and return a list of sequences.
        """
        sequences = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line and re.match("^[ACGTacgt]+$", line):  # only nucleotide lines
                    sequences.append(line.upper())
        return sequences[:self.sample_size]

    def convert_to_biallelic(self, sequences):
        """
        Convert sequences to biallelic 0/1 format.
        Handles sequences of varying lengths by trimming to shortest.
        """
        # Find shortest sequence length
        min_len = min(len(seq) for seq in sequences)
        # Trim all sequences to shortest
        seq_array = np.array([list(seq[:min_len]) for seq in sequences])
        
        n_sites = seq_array.shape[1]
        biallelic_matrix = np.zeros_like(seq_array, dtype=int)

        for i in range(n_sites):
            column = seq_array[:, i]
            counts = Counter(column)
            if len(counts) == 1:
                # all identical
                biallelic_matrix[:, i] = 0
            else:
                major, _ = counts.most_common(1)[0]
                for j, base in enumerate(column):
                    biallelic_matrix[j, i] = 0 if base == major else 1
        return biallelic_matrix

    def save_matrix(self, matrix, out_file):
        """
        Save the biallelic matrix as a text file.
        """
        np.savetxt(out_file, matrix, fmt="%d")

# Main
if __name__ == "__main__":
    replicate_folder = "simulation replicates"
    out_folder = "biallelic matrices"
    os.makedirs(out_folder, exist_ok=True)

    converter = BiallelicConverter(sample_size=100)

    for r in range(1, 11):
        in_file = os.path.join(replicate_folder, f"replicate_{r}.fasta")
        if not os.path.exists(in_file):
            print(f"File {in_file} not found. Skipping replicate {r}.")
            continue

        sequences = converter.read_sequences(in_file)
        if not sequences:
            print(f"No sequences found in replicate {r}. Skipping.")
            continue

        matrix = converter.convert_to_biallelic(sequences)
        out_file = os.path.join(out_folder, f"01_matrix_{r}.txt")
        converter.save_matrix(matrix, out_file)

        print(f"Replicate {r}: biallelic matrix saved to {out_file}")
