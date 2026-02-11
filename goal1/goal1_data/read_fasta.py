# This script reads a FASTA file and prints the header and sequences

fasta_file = "simulation replicates/replicate_8.fasta"

with open(fasta_file, "r") as f:
    lines = f.readlines()

sequence = ""
header = ""

for line in lines:
    line = line.strip()
    if line.startswith(">"):
        header = line
    else:
        sequence += line

print("FASTA Header:")
print(header)
print("\nSequence (first 1000 nt shown):")
print(sequence[:1000])
print(f"\nTotal sequence length: {len(sequence)}")