"""
Chaos Game Representation (CGR) + Triplet Loss Encoder for DNA Sequences
Optimized for limited compute (Colab/Kaggle free tier)
Uses tqdm for all output to avoid console overflow
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import pickle
import sys

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

# ==================== CGR Generation ====================

class CGREncoder:
    """Chaos Game Representation encoder for DNA sequences"""
    
    def __init__(self, resolution=64):
        """
        Args:
            resolution: Size of the CGR image (resolution x resolution)
        """
        self.resolution = resolution
        # Define nucleotide positions (corners of unit square)
        self.positions = {
            'A': np.array([0, 0]),
            'T': np.array([1, 0]),
            'G': np.array([0, 1]),
            'C': np.array([1, 1])
        }
        
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Convert DNA sequence to CGR image
        
        Args:
            sequence: DNA sequence string (ATGC)
            
        Returns:
            CGR image as 2D numpy array (resolution x resolution)
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        sequence = ''.join([c for c in sequence if c in 'ATGC'])
        
        if len(sequence) == 0:
            return np.zeros((self.resolution, self.resolution))
        
        # Initialize position at center
        position = np.array([0.5, 0.5])
        positions_list = [position.copy()]
        
        # Generate CGR coordinates
        for nucleotide in sequence:
            if nucleotide in self.positions:
                corner = self.positions[nucleotide]
                # Move halfway to the corner
                position = (position + corner) / 2
                positions_list.append(position.copy())
        
        # Convert to image
        cgr_image = np.zeros((self.resolution, self.resolution))
        
        for pos in positions_list:
            x = int(pos[0] * (self.resolution - 1))
            y = int(pos[1] * (self.resolution - 1))
            cgr_image[y, x] += 1
        
        # Log transform and normalize
        cgr_image = np.log1p(cgr_image)
        if cgr_image.max() > 0:
            cgr_image = cgr_image / cgr_image.max()
        
        return cgr_image.astype(np.float32)

# ==================== Dataset ====================

# In encomp.py, replace the FASTADataset class:

class FASTADataset(Dataset):
    """
    Dataset for loading PRE-COMPUTED CGR files from a directory.
    It reads FASTA files first to get the sequence IDs and labels,
    then loads the corresponding .npy CGR images.
    """
    
    def __init__(self, fasta_files: List[str], 
                 cgr_preprocessed_dir: str,
                 taxonomy_map: Optional[Dict[str, int]] = None):
        """
        Args:
            fasta_files: List of paths to *original* FASTA files (for IDs/labels)
            cgr_preprocessed_dir: Path to directory containing .npy CGRs
            taxonomy_map: Dict mapping seq_id to INTEGER label
        """
        self.cgr_dir = Path(cgr_preprocessed_dir)
        self.taxonomy_map = taxonomy_map or {}
        
        self.cgr_paths = []
        self.labels = []
        self.seq_ids = []
        
        # We still parse the FASTA files, but *only* for IDs and labels.
        # We don't store the sequences in memory.
        with tqdm(total=len(fasta_files), desc="Loading FASTA metadata", file=sys.stdout) as pbar:
            for fasta_file in fasta_files:
                self._load_metadata_from_fasta(fasta_file)
                pbar.update(1)
        
        tqdm.write(f"✓ Loaded metadata for {len(self.seq_ids)} sequences")

    def _load_metadata_from_fasta(self, fasta_file: str):
        """Load sequence IDs and labels from a FASTA file"""
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    seq_id = line[1:].split()[0]
                    
                    # Clean the seq_id to match the .npy filename
                    clean_seq_id = seq_id.replace('/', '_').replace('|', '_')
                    cgr_path = self.cgr_dir / f"{clean_seq_id}.npy"
                    
                    # Only add if the preprocessed file exists
                    if cgr_path.exists():
                        self.seq_ids.append(seq_id)
                        self.cgr_paths.append(str(cgr_path))
                        
                        # Get taxonomy label if available
                        label = self.taxonomy_map.get(seq_id, -1)
                        self.labels.append(label)
                    else:
                        tqdm.write(f"⚠ Warning: Missing CGR file for {seq_id}")
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        """Get pre-computed CGR image and label for a sequence"""
        
        # THIS IS NOW LIGHTNING FAST I/O:
        cgr = np.load(self.cgr_paths[idx])
        
        # Add channel dimension
        cgr = cgr[np.newaxis, :, :]  # (1, H, W)
        
        label = self.labels[idx]
        
        return {
            'cgr': torch.from_numpy(cgr),
            'label': label,
            'seq_id': self.seq_ids[idx]
        }

# ==================== CNN Encoder ====================

class CGREncoder2D(nn.Module):
    """Lightweight 2D CNN encoder for CGR images"""
    
    def __init__(self, input_size=64, latent_dim=128):
        """
        Args:
            input_size: Size of input CGR image (assumes square)
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Conv block 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size
        self.flat_size = 256 * (input_size // 16) * (input_size // 16)
        
        # Projection to latent space
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input CGR images (B, 1, H, W)
            
        Returns:
            Latent embeddings (B, latent_dim)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalize for better triplet loss performance
        x = F.normalize(x, p=2, dim=1)
        return x

# ==================== Triplet Loss ====================

class TripletLoss(nn.Module):
    """Online triplet mining with hard negative/positive mining"""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) tensor of embeddings
            labels: (B,) tensor of labels
            
        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # For each anchor, find hardest positive and hardest negative
        mask_anchor_positive = self._get_anchor_positive_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_mask(labels)
        
        # Hardest positive: max distance among positives
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # Hardest negative: min distance among negatives
        max_anchor_negative_dist = pairwise_dist.max()
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    def _get_anchor_positive_mask(self, labels):
        """Get mask for valid positive pairs"""
        labels = labels.unsqueeze(0)
        labels_equal = labels.t() == labels
        labels_equal.fill_diagonal_(False)
        return labels_equal.float()
    
    def _get_anchor_negative_mask(self, labels):
        """Get mask for valid negative pairs"""
        labels = labels.unsqueeze(0)
        labels_not_equal = labels.t() != labels
        return labels_not_equal.float()

# ==================== Training ====================

class CGRTripletTrainer:
    """Trainer for CGR encoder with triplet loss"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', use_data_parallel=True):
        # Use DataParallel for multi-GPU
        if use_data_parallel and torch.cuda.device_count() > 1:
            tqdm.write(f"✓ Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss(margin=0.3)
        tqdm.write(f"✓ Model initialized on {device}")
        
    def train(self, train_loader, num_epochs=50, lr=1e-3, save_path='cgr_encoder.pt'):
        """Train the encoder"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Use mixed precision training for faster training on T4s
        scaler = torch.amp.GradScaler(device="cuda")
        
        history = {'loss': []}
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", file=sys.stdout)
        
        for epoch in epoch_pbar:
            self.model.train()
            epoch_loss = 0
            valid_batches = 0
            
            # Batch progress bar
            batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                            leave=False, file=sys.stdout)
            
            for batch in batch_pbar:
                cgr = batch['cgr'].to(self.device, non_blocking=True)
                labels = batch['label']
                
                # Skip batches without valid labels or with single class
                if (labels == -1).all() or len(labels.unique()) == 1:
                    continue
                
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision training
                with torch.amp.autocast(device_type="cuda"):
                    embeddings = self.model(cgr)
                    loss = self.criterion(embeddings, labels)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            batch_pbar.close()
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                history['loss'].append(avg_loss)
                scheduler.step(avg_loss)
                
                current_lr = optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    # Save the underlying model if using DataParallel
                    model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    torch.save(model_to_save.state_dict(), save_path)
                    tqdm.write(f'  → Checkpoint saved (epoch {epoch+1})')
            else:
                tqdm.write(f"\n⚠ Warning: No valid batches in epoch {epoch+1}")
        
        epoch_pbar.close()
        
        # Save final model
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(model_to_save.state_dict(), save_path)
        tqdm.write(f'✓ Training complete. Model saved to {save_path}')
        
        return history
    
    def extract_embeddings(self, dataloader):
        """Extract embeddings for all sequences"""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        all_seq_ids = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Extracting embeddings', file=sys.stdout)
            for batch in pbar:
                cgr = batch['cgr'].to(self.device, non_blocking=True)
                
                # Use mixed precision for inference too
                with torch.amp.autocast(device_type="cuda"):
                    embeddings = self.model(cgr)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_seq_ids.extend(batch['seq_id'])
            pbar.close()
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        tqdm.write(f"✓ Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
        
        return embeddings, labels, all_seq_ids

# ==================== Main Pipeline ====================

def check_gpu_availability():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        tqdm.write(f"✓ Found {num_gpus} GPU(s):")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            tqdm.write(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        tqdm.write("⚠ No GPU found, using CPU")
        return False

def load_taxonomy_map(taxonomy_file: str) -> Dict[str, int]:
    """
    Load taxonomy mapping from file
    Expected format: seq_id\ttax_id (one per line)
    """
    taxonomy_map = {}
    
    with open(taxonomy_file, 'r') as f:
        lines = f.readlines()
    
    with tqdm(total=len(lines), desc="Loading taxonomy", file=sys.stdout) as pbar:
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                seq_id, tax_id = parts[0], parts[1]
                taxonomy_map[seq_id] = tax_id
            pbar.update(1)
    
    # Convert to integer labels
    unique_taxa = sorted(set(taxonomy_map.values()))
    taxa_to_label = {tax: i for i, tax in enumerate(unique_taxa)}
    
    taxonomy_map = {seq_id: taxa_to_label[tax] 
                    for seq_id, tax in taxonomy_map.items()}
    
    tqdm.write(f"✓ Loaded {len(taxonomy_map)} taxonomy entries ({len(unique_taxa)} unique taxa)")
    
    return taxonomy_map

def main():
    """Main pipeline for CGR encoding and latent space extraction"""
    
    tqdm.write("="*60)
    tqdm.write("CGR ENCODER PIPELINE")
    tqdm.write("="*60)
    
    # Configuration
    FASTA_DIR = './data'  # Directory containing .fasta files
    CGR_PREPROCESSED_DIR = './data/cgr_preprocessed'
    TAXONOMY_FILE = './data/taxonomy.txt'  # Optional taxonomy mapping
    CGR_RESOLUTION = 64  # 64x64 CGR images (can reduce to 32 for faster training)
    LATENT_DIM = 128  # Latent space dimension
    BATCH_SIZE = 256  # INCREASED for better GPU utilization (was 32)
    NUM_EPOCHS = 50
    USE_PCA = False  # Set True if you want PCA preprocessing
    PCA_COMPONENTS = 100  # Only used if USE_PCA=True
    
    # Load taxonomy map (optional)
    taxonomy_map = None
    if TAXONOMY_FILE is not None and Path(TAXONOMY_FILE).exists():
        taxonomy_map = load_taxonomy_map(TAXONOMY_FILE)
    else:
        tqdm.write("⚠ No taxonomy file provided, proceeding without labels")
    
    # Get all FASTA files
    fasta_files = list(Path(FASTA_DIR).glob('*.fasta')) + \
                  list(Path(FASTA_DIR).glob('*.fa'))
    
    if len(fasta_files) == 0:
        tqdm.write(f"❌ ERROR: No FASTA files found in {FASTA_DIR}")
        return
    
    tqdm.write(f"✓ Found {len(fasta_files)} FASTA files")
    
    # Create dataset
    tqdm.write("\n" + "="*60)
    tqdm.write("STEP 1: Loading Dataset from Pre-processed CGRs")
    tqdm.write("="*60)
    
    # Check if CGRs exist
    if not Path(CGR_PREPROCESSED_DIR).exists() or \
       not any(Path(CGR_PREPROCESSED_DIR).glob('*.npy')):
        tqdm.write(f"❌ ERROR: Pre-processed CGRs not found in {CGR_PREPROCESSED_DIR}")
        tqdm.write(f"  → Please run the pre-processing script first.")
        return

    dataset = FASTADataset(
        fasta_files=[str(f) for f in fasta_files],
        cgr_preprocessed_dir=CGR_PREPROCESSED_DIR, # <-- PASS NEW DIR
        taxonomy_map=taxonomy_map
    )
    
    # Check if we have any labels
    labels = np.array(dataset.labels)
    has_labels = taxonomy_map is not None and (labels != -1).any()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Increased for better CPU->GPU pipeline
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch more batches
    )
    
    # Option 1: Direct CGR -> Clustering (no deep learning)
    # Use this if you don't have taxonomy labels OR if you want faster results
    if not has_labels:
        tqdm.write("\n" + "="*60)
        tqdm.write("NO LABELS FOUND - USING RAW CGR FEATURES")
        tqdm.write("="*60)
        tqdm.write("Skipping deep learning, extracting raw CGR representations...")
        
        cgr_features = []
        pbar = tqdm(range(len(dataset)), desc="Generating CGR features", file=sys.stdout)
        for i in pbar:
            item = dataset[i]
            cgr = item['cgr'].numpy().flatten()
            cgr_features.append(cgr)
        pbar.close()
        
        cgr_features = np.array(cgr_features)
        tqdm.write(f"✓ CGR feature shape: {cgr_features.shape}")
        
        # Apply PCA if needed (recommended for large CGR features)
        if USE_PCA or cgr_features.shape[1] > 500:
            tqdm.write(f"Applying PCA (components={PCA_COMPONENTS})...")
            pca = PCA(n_components=PCA_COMPONENTS)
            cgr_features = pca.fit_transform(cgr_features)
            explained_var = pca.explained_variance_ratio_.sum()
            tqdm.write(f"✓ PCA reduced shape: {cgr_features.shape}")
            tqdm.write(f"✓ Explained variance: {explained_var*100:.1f}%")
        
        # Save features
        np.save('latent_embeddings.npy', cgr_features)
        np.save('labels.npy', labels)
        with open('seq_ids.pkl', 'wb') as f:
            pickle.dump(dataset.seq_ids, f)
        
        tqdm.write("✓ Features saved to latent_embeddings.npy!")
        tqdm.write("\n" + "="*60)
        tqdm.write("✓ PIPELINE COMPLETE!")
        tqdm.write("="*60)
        tqdm.write("\nNext step: Run hdbscan_classifier.py for clustering")
        return
    
    if False:  # Set to True to skip training and use raw CGR even with labels
        tqdm.write("\n" + "="*60)
        tqdm.write("EXTRACTING RAW CGR (No Training)")
        tqdm.write("="*60)
        
        cgr_features = []
        pbar = tqdm(range(len(dataset)), desc="Generating CGR features", file=sys.stdout)
        for i in pbar:
            item = dataset[i]
            cgr = item['cgr'].numpy().flatten()
            cgr_features.append(cgr)
        pbar.close()
        
        cgr_features = np.array(cgr_features)
        tqdm.write(f"✓ CGR feature shape: {cgr_features.shape}")
        
        # Apply PCA if needed
        if USE_PCA:
            tqdm.write(f"Applying PCA (components={PCA_COMPONENTS})...")
            pca = PCA(n_components=PCA_COMPONENTS)
            cgr_features = pca.fit_transform(cgr_features)
            tqdm.write(f"✓ PCA reduced shape: {cgr_features.shape}")
        
        # Save features
        np.save('cgr_features.npy', cgr_features)
        with open('seq_ids.pkl', 'wb') as f:
            pickle.dump(dataset.seq_ids, f)
        
        tqdm.write("✓ Features saved!")
        return
    
    # Option 2: Train encoder with triplet loss
    tqdm.write("\n" + "="*60)
    tqdm.write("STEP 2: Training Encoder")
    tqdm.write("="*60)
    
    # Create model
    model = CGREncoder2D(input_size=CGR_RESOLUTION, latent_dim=LATENT_DIM)
    param_count = sum(p.numel() for p in model.parameters())
    tqdm.write(f"Model parameters: {param_count:,}")
    
    # Train
    trainer = CGRTripletTrainer(model)
    history = trainer.train(
        dataloader,
        num_epochs=NUM_EPOCHS,
        lr=1e-3,
        save_path='cgr_encoder.pt'
    )
    
    # Extract embeddings
    tqdm.write("\n" + "="*60)
    tqdm.write("STEP 3: Extracting Embeddings")
    tqdm.write("="*60)
    
    embeddings, labels, seq_ids = trainer.extract_embeddings(dataloader)
    
    # Save embeddings
    np.save('latent_embeddings.npy', embeddings)
    np.save('labels.npy', labels)
    with open('seq_ids.pkl', 'wb') as f:
        pickle.dump(seq_ids, f)
    
    tqdm.write("✓ Embeddings saved to latent_embeddings.npy")
    tqdm.write("✓ Labels saved to labels.npy")
    tqdm.write("✓ Sequence IDs saved to seq_ids.pkl")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    tqdm.write("✓ Loss curve saved to training_loss.png")
    
    tqdm.write("\n" + "="*60)
    tqdm.write("✓ PIPELINE COMPLETE!")
    tqdm.write("="*60)
    tqdm.write("\nNext step: Run hdbscan_classifier.py for clustering")

if __name__ == '__main__':
    main()