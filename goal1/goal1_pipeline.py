import os
import numpy as np
import pandas as pd
from ripser import Rips

class Goal1Pipeline:
    """
    Pipeline for Goal 1: Testing Noise and Scarcity Robustness
    
    Features:
    - Loads biallelic matrices from disk
    - Computes Hamming distance matrices
    - Applies scarcity (fractional subsampling) and noise (Gaussian perturbation)
    - Computes persistent homology (H0, H1, H2) using Ripser
    - Outputs CSV files summarizing Betti numbers and barcode statistics
    """

    def __init__(self, folder, simulations=10, maxdim=2):
        """
        Parameters
        ----------
        folder : str
            Folder where the biallelic matrices are stored
        simulations : int
            Number of replicate simulations
        maxdim : int
            Maximum homology dimension (H0-H2)
        """
        self.folder = folder
        self.SIMULATIONS = simulations
        self.MAXDIM = maxdim
        self.ripser = Rips(maxdim=maxdim, verbose=False)

        # Define noise and scarcity levels
        self.varlist = np.around(np.linspace(0, 100, 21), decimals=2).tolist()
        self.spalist = np.around(np.linspace(0.1, 1.0, 10), decimals=2).tolist()

    # Load biallelic matrices
    def load_matrices(self):
        """Load all biallelic matrices (replicates) from folder."""
        matrices = []
        for n in range(1, self.SIMULATIONS + 1):
            fname = os.path.join(self.folder, f"01_matrix_{n}.txt")
            if not os.path.exists(fname):
                print(f"Warning: {fname} not found, skipping replicate {n}")
                continue
            mat = np.loadtxt(fname, dtype=int)
            matrices.append(mat)
        self.matrices = matrices
        return matrices

    # Compute Hamming distance matrix
    def hamming_distance_matrix(self, mat):
        """
        Compute the Hamming distance matrix for a replicate.
        Distance = number of differing positions between sequences.
        """
        n = mat.shape[0]
        hd = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum(mat[i] != mat[j])
                hd[i, j] = d
                hd[j, i] = d
        return hd

    # Add Gaussian noise
    def add_noise(self, hd_matrix, var):
        """
        Adds symmetric Gaussian noise to a Hamming distance matrix.
        Diagonal is kept zero.
        """
        n = hd_matrix.shape[0]
        noise = np.random.normal(loc=0, scale=np.sqrt(var), size=(n, n))
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        return hd_matrix + noise

    # Subsample sequences for sparsity
    def sparsity_sample(self, mat, fraction):
        """
        Randomly sample a fraction of sequences to simulate scarcity.
        """
        n = mat.shape[0]
        k = max(1, int(n * fraction))
        idx = np.random.choice(n, k, replace=False)
        return mat[idx, :][:, idx]

    # Compute persistence diagrams
    def compute_persistence(self, hd_matrix):
        """
        Compute persistent homology up to self.MAXDIM using Ripser.
        """
        hd_matrix = np.array(hd_matrix, dtype=float)
        diagrams = self.ripser.fit_transform(hd_matrix, distance_matrix=True)
        # Sort intervals by death time
        for i in range(self.MAXDIM + 1):
            if diagrams[i].size > 0:
                diagrams[i] = diagrams[i][np.argsort(diagrams[i][:, 1])]
        return diagrams

    # Compute barcode lengths
    def barcode_lengths(self, diagrams):
        """Compute lengths (death - birth) for each homology group."""
        lengths = []
        for i in range(self.MAXDIM + 1):
            if diagrams[i].size > 0:
                lengths.append(diagrams[i][:, 1] - diagrams[i][:, 0])
            else:
                lengths.append(np.array([]))
        return lengths

    # Compute Betti numbers
    def betti_numbers(self, diagrams):
        """Count number of intervals per homology dimension."""
        return [len(diagrams[i]) for i in range(self.MAXDIM + 1)]

    # Pipeline: Scarcity analysis
    def pipeline_scarcity(self):
        """Save full persistence intervals for scarcity levels."""
        rows = []

        for rep_idx, mat in enumerate(self.matrices):
            for frac in self.spalist:

                if mat.shape[0] < 2:
                    continue

                mat_sparse = self.sparsity_sample(mat, frac)
                hd_sparse = self.hamming_distance_matrix(mat_sparse)
                diagrams = self.compute_persistence(hd_sparse)

                for dim in range(self.MAXDIM + 1):
                    for birth, death in diagrams[dim]:

                        # Remove infinite deaths
                        if not np.isfinite(death):
                            continue

                        rows.append({
                            "replicate": rep_idx + 1,
                            "level_type": "scarcity",
                            "level_value": frac,
                            "dimension": dim,
                            "birth": birth,
                            "death": death,
                            "length": death - birth
                        })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.folder, "scarcity_levels.csv"), index=False)
        print("Saved scarcity_levels.csv")

    # Pipeline: Noise analysis
    def pipeline_noise(self):
        """Save full persistence intervals for noise levels."""
        rows = []

        for rep_idx, mat in enumerate(self.matrices):

            if mat.shape[0] < 2:
                continue

            hd = self.hamming_distance_matrix(mat)

            for var in self.varlist:
                hd_noisy = self.add_noise(hd, var)
                diagrams = self.compute_persistence(hd_noisy)

                for dim in range(self.MAXDIM + 1):
                    for birth, death in diagrams[dim]:

                        if not np.isfinite(death):
                            continue

                        rows.append({
                            "replicate": rep_idx + 1,
                            "level_type": "noise",
                            "level_value": var,
                            "dimension": dim,
                            "birth": birth,
                            "death": death,
                            "length": death - birth
                        })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.folder, "noise_levels.csv"), index=False)
        print("Saved noise_levels.csv")

# Usage
folder = "goal1_data/biallelic matrices"  # folder containing 01_matrix_1.txt, 01_matrix_2.txt, ...
pipeline = Goal1Pipeline(folder, simulations=10)

pipeline.load_matrices()

pipeline.pipeline_scarcity()

pipeline.pipeline_noise()
