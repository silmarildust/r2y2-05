import random

def generate_nucleotide(gc_content=0.5):
    """
    Generate a nucleotide randomly based on GC content.
    gc_content: fraction of G/C in the sequence (0-1)
    """
    if random.random() < gc_content:
        return random.choice("GC")
    else:
        return random.choice("AT")

def generate_sequence(length=1300, gc_content=0.5):
    """
    Generate a single viral-like DNA sequence of given length and GC content
    """
    return ''.join(generate_nucleotide(gc_content) for _ in range(length))

def generate_sequences(num_sequences=10, length=1300, gc_content=0.5):
    """
    Generate multiple sequences
    """
    sequences = []
    for i in range(num_sequences):
        seq = generate_sequence(length, gc_content)
        sequences.append(seq)
    return sequences

def save_fasta(sequences, filename="viral_sequences.fasta"):
    """
    Save sequences in FASTA format
    """
    with open(filename, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n")
            # wrap every 70 chars per FASTA convention
            for j in range(0, len(seq), 70):
                f.write(seq[j:j+70] + "\n")
    print(f"{len(sequences)} sequences saved to {filename}")

# --------- MAIN ---------
if __name__ == "__main__":
    NUM_SEQUENCES = 10       # how many sequences you want
    SEQ_LENGTH = 1302        # length of each sequence
    GC_CONTENT = 0.45        # realistic viral GC content (~40-50%)

    sequences = generate_sequences(NUM_SEQUENCES, SEQ_LENGTH, GC_CONTENT)
    save_fasta(sequences)