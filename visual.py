"""
Visualization and validation tools for CGR pipeline
Run this before training to check your data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from collections import Counter

# Import from your cgr_encoder.py
from encomp import CGREncoder, FASTADataset, load_taxonomy_map

def visualize_cgr_samples(dataset, num_samples=9, save_path='cgr_samples.png'):
    """
    Visualize random CGR representations
    
    Args:
        dataset: FASTADataset instance
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in zip(indices, axes):
        item = dataset[idx]
        cgr = item['cgr'].squeeze().numpy()
        seq_id = item['seq_id']
        label = item['label']
        
        ax.imshow(cgr, cmap='viridis', interpolation='nearest')
        ax.set_title(f'ID: {seq_id[:20]}\nLabel: {label}', fontsize=8)
        ax.axis('off')
    
    # Hide extra subplots
    for ax in axes[len(indices):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CGR visualization saved to {save_path}")

def compare_sequences_cgr(sequences, labels=None, save_path='cgr_comparison.png'):
    """
    Compare CGR representations of specific sequences
    
    Args:
        sequences: List of DNA sequences (strings)
        labels: Optional list of labels for sequences
        save_path: Path to save comparison
    """
    cgr_encoder = CGREncoder(resolution=64)
    
    if labels is None:
        labels = [f"Seq {i+1}" for i in range(len(sequences))]
    
    fig, axes = plt.subplots(1, len(sequences), figsize=(4*len(sequences), 4))
    if len(sequences) == 1:
        axes = [axes]
    
    for seq, label, ax in zip(sequences, labels, axes):
        cgr = cgr_encoder.encode_sequence(seq)
        
        im = ax.imshow(cgr, cmap='viridis', interpolation='nearest')
        ax.set_title(f'{label}\nLength: {len(seq)}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CGR comparison saved to {save_path}")

def analyze_dataset(dataset):
    """
    Analyze and print statistics about the dataset
    
    Args:
        dataset: FASTADataset instance
    """
    print("\n" + "="*60)
    print("Dataset Analysis")
    print("="*60)
    
    # Basic stats
    print(f"\nTotal sequences: {len(dataset)}")
    
    # Sequence length distribution
    lengths = [len(seq) for seq in dataset.seq_ids]
    print(f"\nSequence lengths:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    
    # Plot length distribution
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Sequence Lengths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sequence_lengths.png')
    plt.close()
    print("  Length distribution saved to sequence_lengths.png")
    
    # Label distribution
    label_counts = Counter(dataset.labels)
    print(f"\nLabel distribution:")
    print(f"  Unique labels: {len(label_counts)}")
    print(f"  Unknown (-1): {label_counts.get(-1, 0)}")
    
    # Show label distribution
    if len(label_counts) > 1:
        labels_sorted = sorted([(k, v) for k, v in label_counts.items() if k != -1], 
                              key=lambda x: x[1], reverse=True)
        
        print("\n  Top 10 labels by count:")
        for label, count in labels_sorted[:10]:
            print(f"    Label {label}: {count} sequences")
        
        # Plot label distribution
        if len(labels_sorted) > 0:
            plot_labels, plot_counts = zip(*labels_sorted[:20])
            
            plt.figure(figsize=(12, 5))
            plt.bar(range(len(plot_labels)), plot_counts)
            plt.xlabel('Label ID')
            plt.ylabel('Count')
            plt.title('Top 20 Label Distribution')
            plt.xticks(range(len(plot_labels)), plot_labels, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('label_distribution.png')
            plt.close()
            print("  Label distribution saved to label_distribution.png")
    
    # Check nucleotide composition
    all_nucleotides = ''.join(dataset.seq_ids)
    nuc_counts = Counter(all_nucleotides)
    total = sum(nuc_counts.values())
    
    print(f"\nNucleotide composition:")
    for nuc in ['A', 'T', 'G', 'C']:
        count = nuc_counts.get(nuc, 0) + nuc_counts.get(nuc.lower(), 0)
        pct = 100 * count / total
        print(f"  {nuc}: {pct:.2f}%")
    
    # Check for non-standard nucleotides
    non_standard = {k: v for k, v in nuc_counts.items() 
                   if k.upper() not in 'ATGC'}
    if non_standard:
        print(f"\n  Warning: Non-standard nucleotides found:")
        for nuc, count in non_standard.items():
            print(f"    '{nuc}': {count} occurrences")

def test_cgr_properties():
    """
    Test and visualize key properties of CGR encoding
    """
    print("\n" + "="*60)
    print("Testing CGR Properties")
    print("="*60)
    
    # Test 1: Similar sequences should have similar CGRs
    print("\n1. Testing similarity preservation...")
    
    base_seq = "ATCGATCGATCG" * 10
    similar_seq = "ATCGATCGAT" + "CG" * 10  # Slight variation
    different_seq = "AAAAAAAAAAAAGGGGGGGGGGGGCCCCCCCCCCCC"
    
    compare_sequences_cgr(
        [base_seq, similar_seq, different_seq],
        ['Base', 'Similar', 'Different'],
        'cgr_similarity_test.png'
    )
    print("  ✓ Similarity test visualization saved")
    
    # Test 2: Effect of sequence length
    print("\n2. Testing effect of sequence length...")
    
    short_seq = "ATCG" * 5
    medium_seq = "ATCG" * 20
    long_seq = "ATCG" * 100
    
    compare_sequences_cgr(
        [short_seq, medium_seq, long_seq],
        ['Short (20bp)', 'Medium (80bp)', 'Long (400bp)'],
        'cgr_length_test.png'
    )
    print("  ✓ Length test visualization saved")
    
    # Test 3: Different composition
    print("\n3. Testing different nucleotide compositions...")
    
    at_rich = "ATATAT" * 20
    gc_rich = "GCGCGC" * 20
    balanced = "ATCGATCG" * 15
    
    compare_sequences_cgr(
        [at_rich, gc_rich, balanced],
        ['AT-rich', 'GC-rich', 'Balanced'],
        'cgr_composition_test.png'
    )
    print("  ✓ Composition test visualization saved")

def validate_pipeline(fasta_dir, taxonomy_file=None, num_test_samples=5):
    """
    Complete validation of the pipeline before training
    
    Args:
        fasta_dir: Directory with FASTA files
        taxonomy_file: Optional taxonomy mapping file
        num_test_samples: Number of samples to visualize
    """
    print("\n" + "="*60)
    print("PIPELINE VALIDATION")
    print("="*60)
    
    # Check files
    fasta_files = list(Path(fasta_dir).glob('*.fasta')) + \
                  list(Path(fasta_dir).glob('*.fa'))
    
    if len(fasta_files) == 0:
        print(f"❌ ERROR: No FASTA files found in {fasta_dir}")
        return False
    
    print(f"\n✓ Found {len(fasta_files)} FASTA files")
    
    # Load taxonomy if available
    taxonomy_map = None
    if taxonomy_file and Path(taxonomy_file).exists():
        taxonomy_map = load_taxonomy_map(taxonomy_file)
        print(f"✓ Loaded taxonomy map with {len(taxonomy_map)} entries")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = FASTADataset(
            fasta_files=[str(f) for f in fasta_files],
            cgr_preprocessed_dir='./data/cgr_preprocessed',
            taxonomy_map=taxonomy_map
        )
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading dataset: {e}")
        return False
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Visualize samples
    print("\nGenerating visualizations...")
    visualize_cgr_samples(dataset, num_samples=min(9, len(dataset)))
    print("✓ Sample visualizations created")
    
    # Test CGR properties
    test_cgr_properties()
    
    # Memory estimate
    cgr_memory = len(dataset) * 64 * 64 * 4 / (1024**2)  # MB
    print(f"\nMemory estimate:")
    print(f"  CGR images: ~{cgr_memory:.1f} MB")
    print(f"  Embeddings (128D): ~{len(dataset) * 128 * 4 / (1024**2):.1f} MB")
    
    print("\n" + "="*60)
    print("✓ VALIDATION COMPLETE - Ready to train!")
    print("="*60)
    
    return True

def quick_check(fasta_file):
    """
    Quick check of a single FASTA file
    
    Args:
        fasta_file: Path to FASTA file
    """
    print(f"\nQuick check of {fasta_file}...")
    
    sequences = []
    seq_ids = []
    
    with open(fasta_file, 'r') as f:
        seq_id = None
        sequence = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_id and sequence:
                    sequences.append(''.join(sequence))
                    seq_ids.append(seq_id)
                seq_id = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)
        
        if seq_id and sequence:
            sequences.append(''.join(sequence))
            seq_ids.append(seq_id)
    
    print(f"  Found {len(sequences)} sequences")
    
    if len(sequences) > 0:
        print(f"  First sequence ID: {seq_ids[0]}")
        print(f"  First sequence length: {len(sequences[0])}")
        print(f"  First 50 bases: {sequences[0][:50]}")
        
        # Visualize first few
        compare_sequences_cgr(
            sequences[:3],
            seq_ids[:3],
            'quick_check_cgr.png'
        )

# ==================== Main ====================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Full validation: python cgr_visualizer.py /path/to/fasta/dir [taxonomy.txt]")
        print("  Quick check: python cgr_visualizer.py /path/to/file.fasta --quick")
        sys.exit(1)
    
    fasta_path = sys.argv[1]
    
    if '--quick' in sys.argv or Path(fasta_path).is_file():
        # Quick check mode
        quick_check(fasta_path)
    else:
        # Full validation mode
        taxonomy_file = sys.argv[2] if len(sys.argv) > 2 else None
        validate_pipeline(fasta_path, taxonomy_file)