from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
from Bio.Align import substitution_matrices

def compare_sequences(seq1, seq2):
    """
    Compare two DNA sequences and return alignment information.
    
    Args:
        seq1 (str): First DNA sequence
        seq2 (str): Second DNA sequence
        
    Returns:
        dict: Dictionary containing alignment score and formatted alignment
    """
    # Convert to uppercase to ensure case insensitivity
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    
    # Load a substitution matrix for DNA
    matrix = substitution_matrices.load("NUC.4.4")
    
    # Perform global alignment
    alignments = pairwise2.align.globalds(
        seq1, seq2,
        matrix,  # scoring matrix
        -10,     # gap open penalty
        -0.5,    # gap extension penalty
        one_alignment_only=True
    )
    
    if not alignments:
        return {"score": 0, "alignment": "No alignment possible"}
    
    # Get the best alignment
    best_alignment = alignments[0]
    
    return {
        "score": best_alignment.score,
        "alignment": format_alignment(*best_alignment, full_sequences=True)
    }

def plot_sequence_comparison(seq1, seq2):
    """
    Create a simple visualization of sequence comparison.
    """
    # Create a simple dot plot
    dots = [(i, j) for i, x in enumerate(seq1) 
            for j, y in enumerate(seq2) if x == y]
    
    if dots:
        x, y = zip(*dots)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=1, alpha=0.5)
        plt.title('Sequence Comparison Dot Plot')
        plt.xlabel('Sequence 1 Position')
        plt.ylabel('Sequence 2 Position')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sequence_comparison.png')
        plt.close()
        return 'sequence_comparison.png'
    return None

if __name__ == "__main__":
    # Example sequences (replace with your actual sequences)
    sequence1 = "ATCGATCGATCGATCGATCG"
    sequence2 = "ATCGATGGATCGATAGATCG"
    
    print("Sequence 1:", sequence1)
    print("Sequence 2:", sequence2)
    
    # Compare sequences
    result = compare_sequences(sequence1, sequence2)
    print("\nAlignment Score:", result["score"])
    print("\nAlignment:")
    print(result["alignment"])
    
    # Generate and save visualization
    plot_file = plot_sequence_comparison(sequence1, sequence2)
    if plot_file:
        print(f"\nVisualization saved as {plot_file}")
