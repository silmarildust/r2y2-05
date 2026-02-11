import random

# Number of seed files to generate
num_files = 10
# Length of each genome
length = 2501

# Nucleotides (A, C, G, T)
nucleotides = ["A", "C", "G", "T"]

for i in range(1, num_files + 1):
    # Generate a random sequence
    sequence = "".join(random.choices(nucleotides, k=length))
    
    # Define the file name
    filename = f"base_seed_{i}.fasta"
    
    # Write to FASTA
    with open(filename, "w") as f:
        f.write(f">seed_sequence_{i}\n")
        f.write(sequence + "\n")
    
    print(f"File '{filename}' created with a realistic 2501-nucleotide sequence.")