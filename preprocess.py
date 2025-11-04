import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import sys

# Import CGREncoder and FASTADataset (or just the FASTA parser)
from encomp import CGREncoder, FASTADataset 

# --- Configuration ---
FASTA_DIR = './data'
CGR_RESOLUTION = 64
OUTPUT_DIR = Path('./data/cgr_preprocessed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------

print(f"Starting CGR pre-processing...")
print(f"Output directory: {OUTPUT_DIR}")

# 1. Get all FASTA files
fasta_files = list(Path(FASTA_DIR).glob('*.fasta')) + \
              list(Path(FASTA_DIR).glob('*.fa'))

if not fasta_files:
    print(f"❌ ERROR: No FASTA files found in {FASTA_DIR}")
    sys.exit()

# 2. Use your existing dataset to load sequences (without taxonomy)
# We use this just to parse the FASTA files easily.
dataset = FASTADataset(
    fasta_files=[str(f) for f in fasta_files],
    cgr_resolution=CGR_RESOLUTION,
    taxonomy_map=None
)

cgr_encoder = CGREncoder(resolution=CGR_RESOLUTION)

# 3. Iterate and save CGRs
print(f"Found {len(dataset)} sequences. Generating and saving CGRs...")
pbar = tqdm(total=len(dataset), desc="Pre-processing CGRs", file=sys.stdout)

for i in range(len(dataset)):
    # Get sequence and ID from the dataset
    seq_id = dataset.seq_ids[i]
    sequence = dataset.sequences[i]
    
    # Generate CGR
    cgr_image = cgr_encoder.encode_sequence(sequence)
    
    # Define output path
    # Use a clean filename for the seq_id
    clean_seq_id = seq_id.replace('/', '_').replace('|', '_') 
    output_path = OUTPUT_DIR / f"{clean_seq_id}.npy"
    
    # Save as .npy file
    np.save(output_path, cgr_image)
    pbar.update(1)

pbar.close()
print(f"✓ Pre-processing complete. CGRs saved to {OUTPUT_DIR}")