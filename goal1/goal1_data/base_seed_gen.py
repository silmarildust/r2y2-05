import random

# Number of seed files to generate
num_files = 10
# Length of each genome
length = 2501

nucleotides = ["A", "C", "G", "T"]

for i in range(1, num_files + 1):
    sequence = "".join(random.choices(nucleotides, k=length))
    
    filename = f"base_seed_{i}.fasta"
    
    with open(filename, "w") as f:
        f.write(f">seed_sequence_{i}\n")
        f.write(sequence + "\n")
    

    print(f"File '{filename}' created.")
