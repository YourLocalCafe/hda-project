"""
Extract taxonomy information from FASTA headers
Supports multiple common formats
"""

from pathlib import Path
from tqdm.auto import tqdm
import re
import sys

def extract_from_ncbi_format(header):
    """
    Extract from NCBI format:
    >gi|123456|ref|NC_000001.1| Homo sapiens chromosome 1
    Returns: species name or accession
    """
    # Try to get species name
    species_match = re.search(r'\[([^\]]+)\]', header)
    if species_match:
        return species_match.group(1)
    
    # Try to get accession
    acc_match = re.search(r'\|([A-Z]{2}_\d+)', header)
    if acc_match:
        return acc_match.group(1)
    
    return None

def extract_from_taxid_format(header):
    """
    Extract from headers with TaxID:
    >seq123_taxid_9606_Homo_sapiens
    >seq456|taxid:9606|
    """
    # Format 1: taxid_NUMBER
    match = re.search(r'taxid[_:](\d+)', header, re.IGNORECASE)
    if match:
        return f"taxid_{match.group(1)}"
    
    return None

def extract_from_organism_field(header):
    """
    Extract organism name from various formats:
    >seq123 [organism=Escherichia coli]
    >seq456 organism:Bacillus subtilis
    """
    # Format 1: [organism=NAME]
    match = re.search(r'\[organism=([^\]]+)\]', header, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Format 2: organism:NAME
    match = re.search(r'organism:([^\s]+(?:\s+[^\s]+)?)', header, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def extract_from_genus_species(header):
    """
    Extract genus/species from headers:
    >seq123 Escherichia_coli strain K12
    >seq456_Bacillus_subtilis_genome
    """
    # Look for genus_species pattern (two capitalized words)
    match = re.search(r'[A-Z][a-z]+[_\s][a-z]+', header)
    if match:
        return match.group(0).replace(' ', '_')
    
    return None

def extract_from_custom_delimiter(header, delimiter='|', field_index=1):
    """
    Extract from custom delimited format:
    >seq123|Ecoli|gene1|...
    """
    parts = header.split(delimiter)
    if len(parts) > field_index:
        return parts[field_index].strip()
    return None

def auto_detect_format(fasta_files, sample_size=100):
    """
    Automatically detect the header format by sampling
    """
    tqdm.write("Analyzing FASTA header formats...")
    
    sample_headers = []
    for fasta_file in fasta_files[:5]:  # Check first 5 files
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    sample_headers.append(line.strip())
                    if len(sample_headers) >= sample_size:
                        break
                if len(sample_headers) >= sample_size:
                    break
    
    # Test different extraction methods
    methods = {
        'ncbi': extract_from_ncbi_format,
        'taxid': extract_from_taxid_format,
        'organism': extract_from_organism_field,
        'genus_species': extract_from_genus_species,
    }
    
    results = {}
    for name, method in methods.items():
        success = sum(1 for h in sample_headers if method(h) is not None)
        results[name] = success / len(sample_headers)
    
    tqdm.write("\nFormat detection results:")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        tqdm.write(f"  {name:20s}: {score*100:5.1f}% success rate")
    
    best_method = max(results, key=results.get)
    if results[best_method] > 0.5:
        tqdm.write(f"\n✓ Auto-detected format: {best_method}")
        return methods[best_method]
    else:
        tqdm.write("\n⚠ Could not auto-detect format reliably")
        return None

def create_taxonomy_file(fasta_dir, output_file='taxonomy.txt', 
                        extraction_method='auto', custom_regex=None):
    """
    Create taxonomy mapping file from FASTA headers
    
    Args:
        fasta_dir: Directory with FASTA files
        output_file: Output taxonomy file
        extraction_method: 'auto', 'ncbi', 'taxid', 'organism', 'genus_species', or 'custom'
        custom_regex: If extraction_method='custom', provide regex with one capture group
    """
    tqdm.write("="*60)
    tqdm.write("TAXONOMY EXTRACTION")
    tqdm.write("="*60)
    
    # Get FASTA files
    fasta_files = list(Path(fasta_dir).glob('*.fasta')) + \
                  list(Path(fasta_dir).glob('*.fa'))
    
    if len(fasta_files) == 0:
        tqdm.write(f"❌ ERROR: No FASTA files found in {fasta_dir}")
        return
    
    tqdm.write(f"✓ Found {len(fasta_files)} FASTA files")
    
    # Determine extraction method
    if extraction_method == 'auto':
        extract_func = auto_detect_format(fasta_files)
        if extract_func is None:
            tqdm.write("\n❌ Could not determine format automatically")
            tqdm.write("Please specify extraction_method manually")
            return
    elif extraction_method == 'custom' and custom_regex:
        def extract_func(header):
            match = re.search(custom_regex, header)
            return match.group(1) if match else None
    else:
        method_map = {
            'ncbi': extract_from_ncbi_format,
            'taxid': extract_from_taxid_format,
            'organism': extract_from_organism_field,
            'genus_species': extract_from_genus_species,
        }
        extract_func = method_map.get(extraction_method)
        if extract_func is None:
            tqdm.write(f"❌ Unknown extraction method: {extraction_method}")
            return
    
    # Extract taxonomy from all files
    tqdm.write(f"\nExtracting taxonomy information...")
    taxonomy_map = {}
    failed_count = 0
    
    for fasta_file in tqdm(fasta_files, desc="Processing files", file=sys.stdout):
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    header = line.strip()
                    seq_id = header[1:].split()[0]  # First field after >
                    
                    taxonomy = extract_func(header)
                    
                    if taxonomy:
                        taxonomy_map[seq_id] = taxonomy
                    else:
                        failed_count += 1
    
    tqdm.write(f"\n✓ Successfully extracted: {len(taxonomy_map)} sequences")
    tqdm.write(f"⚠ Failed to extract: {failed_count} sequences")
    
    if len(taxonomy_map) == 0:
        tqdm.write("\n❌ ERROR: No taxonomy information could be extracted!")
        tqdm.write("Your FASTA headers might need a custom extraction pattern")
        return
    
    # Show statistics
    unique_taxa = len(set(taxonomy_map.values()))
    tqdm.write(f"✓ Found {unique_taxa} unique taxa")
    
    from collections import Counter
    taxa_counts = Counter(taxonomy_map.values())
    tqdm.write("\nTop 10 most common taxa:")
    for taxa, count in taxa_counts.most_common(10):
        tqdm.write(f"  {taxa:40s}: {count:5d} sequences")
    
    # Write output file
    tqdm.write(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        for seq_id, taxonomy in sorted(taxonomy_map.items()):
            f.write(f"{seq_id}\t{taxonomy}\n")
    
    tqdm.write(f"✓ Taxonomy file saved!")
    tqdm.write("\n" + "="*60)
    tqdm.write("You can now use this file with cgr_encoder.py:")
    tqdm.write(f"  TAXONOMY_FILE = '{output_file}'")
    tqdm.write("="*60)

def inspect_headers(fasta_file, num_samples=20):
    """
    Show sample headers to help determine extraction pattern
    """
    tqdm.write(f"\nInspecting headers from {fasta_file}...")
    tqdm.write("="*60)
    
    with open(fasta_file, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('>'):
                tqdm.write(line.strip())
                count += 1
                if count >= num_samples:
                    break
    
    tqdm.write("="*60)
    tqdm.write("\nBased on these headers, choose extraction method:")
    tqdm.write("  'ncbi'         - NCBI format with gi|ref|accession")
    tqdm.write("  'taxid'        - Contains taxid_NUMBER or taxid:NUMBER")
    tqdm.write("  'organism'     - Contains [organism=...] or organism:")
    tqdm.write("  'genus_species'- Contains Genus_species pattern")
    tqdm.write("  'custom'       - Provide your own regex pattern")

# ==================== Main ====================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        tqdm.write("Usage:")
        tqdm.write("  Inspect headers:")
        tqdm.write("    python extract_taxonomy.py /path/to/file.fasta --inspect")
        tqdm.write("\n  Auto-extract (recommended):")
        tqdm.write("    python extract_taxonomy.py /path/to/fasta/dir")
        tqdm.write("\n  Manual method:")
        tqdm.write("    python extract_taxonomy.py /path/to/fasta/dir --method ncbi")
        tqdm.write("    python extract_taxonomy.py /path/to/fasta/dir --method taxid")
        tqdm.write("    python extract_taxonomy.py /path/to/fasta/dir --method organism")
        tqdm.write("\n  Custom regex:")
        tqdm.write("    python extract_taxonomy.py /path/to/fasta/dir --method custom --regex 'pattern'")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if '--inspect' in sys.argv:
        # Inspect mode
        inspect_headers(path, num_samples=20)
    else:
        # Extraction mode
        method = 'auto'
        custom_regex = None
        
        if '--method' in sys.argv:
            method_idx = sys.argv.index('--method') + 1
            method = sys.argv[method_idx]
        
        if '--regex' in sys.argv:
            regex_idx = sys.argv.index('--regex') + 1
            custom_regex = sys.argv[regex_idx]
        
        output_file = 'taxonomy.txt'
        if '--output' in sys.argv:
            output_idx = sys.argv.index('--output') + 1
            output_file = sys.argv[output_idx]
        
        create_taxonomy_file(path, output_file, method, custom_regex)