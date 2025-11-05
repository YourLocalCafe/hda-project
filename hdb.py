"""
HDBSCAN Clustering + Taxa Classification for Sequence Embeddings
Downstream pipeline after CGR encoding
Uses tqdm for all output to avoid console overflow
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm.auto import tqdm
from collections import Counter
import sys

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

# ==================== Clustering ====================

class FastClusterer:
    """Fast alternative clustering using MiniBatchKMeans or Agglomerative"""
    
    def __init__(self, n_clusters=100, method='minibatch'):
        """
        Args:
            n_clusters: Number of clusters (estimate)
            method: 'minibatch' or 'agglomerative'
        """
        from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
        
        self.method = method
        self.n_clusters = n_clusters
        
        if method == 'minibatch':
            self.clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                max_iter=100,
                n_init=3,
                random_state=42,
                verbose=1
            )
        else:
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        
        self.scaler = StandardScaler()
    
    def fit(self, embeddings):
        """Fit fast clustering"""
        tqdm.write(f"Standardizing {len(embeddings)} embeddings...")
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        tqdm.write(f"Running {self.method} clustering with {self.n_clusters} clusters...")
        
        import time
        start_time = time.time()
        
        cluster_labels = self.clusterer.fit_predict(embeddings_scaled)
        
        elapsed = time.time() - start_time
        tqdm.write(f"✓ Clustering completed in {elapsed:.1f} seconds")
        
        # Get cluster statistics
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters)
        
        tqdm.write(f"✓ Found {n_clusters} clusters")
        
        cluster_counts = Counter(cluster_labels)
        tqdm.write("\nTop 20 cluster sizes:")
        for cluster_id, count in cluster_counts.most_common(20):
            tqdm.write(f"  Cluster {cluster_id:3d}: {count:5d} sequences")
        
        return cluster_labels

class SequenceClusterer:
    """HDBSCAN-based clustering for sequence embeddings"""
    
    def __init__(self, min_cluster_size=5, min_samples=3, n_jobs=-1):
        """
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples for core points
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        # Set environment variable for parallel processing
        import os
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() if n_jobs == -1 else n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() if n_jobs == -1 else n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() if n_jobs == -1 else n_jobs)
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=n_jobs,  # Parallel distance computation
            algorithm='best',  # Let HDBSCAN choose best algorithm
            approx_min_span_tree=True  # Faster approximate MST
        )
        self.scaler = StandardScaler()
        
    def fit(self, embeddings):
        """
        Fit HDBSCAN on embeddings
        
        Args:
            embeddings: (N, D) array of embeddings
            
        Returns:
            cluster_labels: (N,) array of cluster assignments (-1 for noise)
        """
        tqdm.write(f"Standardizing {len(embeddings)} embeddings...")
        
        # Suppress sklearn warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        tqdm.write(f"Running HDBSCAN clustering (this may take a while for {len(embeddings)} samples)...")
        tqdm.write(f"  Using min_cluster_size={self.clusterer.min_cluster_size}")
        tqdm.write(f"  Using min_samples={self.clusterer.min_samples}")
        tqdm.write(f"  Using approx_min_span_tree=True (faster)")
        
        # HDBSCAN doesn't have built-in progress, so we'll time it
        import time
        start_time = time.time()
        
        cluster_labels = self.clusterer.fit_predict(embeddings_scaled)
        
        elapsed = time.time() - start_time
        tqdm.write(f"✓ Clustering completed in {elapsed:.1f} seconds")
        
        # Get cluster statistics
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters - {-1})
        n_noise = np.sum(cluster_labels == -1)
        
        tqdm.write(f"✓ Found {n_clusters} clusters")
        tqdm.write(f"✓ Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")
        
        # Show cluster sizes
        cluster_counts = Counter(cluster_labels)
        tqdm.write("\nCluster size distribution:")
        for cluster_id in sorted(cluster_counts.keys())[:20]:  # Show first 20
            if cluster_id != -1:
                tqdm.write(f"  Cluster {cluster_id:3d}: {cluster_counts[cluster_id]:5d} sequences")
        if len([c for c in cluster_counts.keys() if c != -1]) > 20:
            tqdm.write(f"  ... and {len([c for c in cluster_counts.keys() if c != -1]) - 20} more clusters")
        
        return cluster_labels
    
    def plot_cluster_distribution(self, cluster_labels, save_path='cluster_distribution.png'):
        """Plot distribution of cluster sizes"""
        cluster_counts = Counter(cluster_labels)
        
        # Remove noise
        if -1 in cluster_counts:
            del cluster_counts[-1]
        
        clusters = sorted(cluster_counts.keys())
        sizes = [cluster_counts[c] for c in clusters]
        
        plt.figure(figsize=(12, 5))
        plt.bar(clusters, sizes)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Sequences')
        plt.title('Cluster Size Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        tqdm.write(f"✓ Cluster distribution plot saved to {save_path}")

# ==================== Classification Network ====================

class ClusterClassifier(nn.Module):
    """Neural network classifier for known/unknown taxa classification"""
    
    def __init__(self, input_dim, num_classes):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of known taxa classes
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings (B, D)
            
        Returns:
            Logits (B, num_classes)
        """
        return self.classifier(x)
    
class TaxaClassifier:
    """Trainer for taxa classification"""
    
    def __init__(self, input_dim, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = ClusterClassifier(input_dim, num_classes).to(device)
        self.device = device
        self.num_classes = num_classes
        self.index_to_taxa = None  # <-- ADD THIS LINE
        tqdm.write(f"✓ Classifier initialized on {device}")
        
    def prepare_data(self, embeddings, labels, test_size=0.2):
        """
        Split data into train/test sets
        
        Args:
            embeddings: (N, D) array of embeddings
            labels: (N,) array of taxa labels
            test_size: Fraction for test set (default 0.2)
            
        Returns:
            train_loader, test_loader
        """
        # Filter out unknown sequences (label=-1)
        mask_known = labels != -1
        embeddings_known = embeddings[mask_known]
        labels_known = labels[mask_known]
        
        tqdm.write(f"Found {len(embeddings_known)} labeled sequences.")

        # --- FIX 1: Filter Singletons ---
        from collections import Counter
        label_counts = Counter(labels_known)
        multi_member_labels = {label for label, count in label_counts.items() if count > 1}
        
        if not multi_member_labels:
            tqdm.write("Error: No classes have more than 1 member. Cannot stratify.")
            return None, None

        mask_multi_member = np.isin(labels_known, list(multi_member_labels))
        embeddings_stratifiable = embeddings_known[mask_multi_member]
        labels_stratifiable = labels_known[mask_multi_member]
        
        n_samples_strat = len(embeddings_stratifiable)
        n_classes_strat = len(multi_member_labels)
        
        tqdm.write(f"Using {n_samples_strat} sequences from {n_classes_strat} classes (with >1 member) for stratified split.")
        tqdm.write(f"  ({(mask_multi_member == False).sum()} singleton sequences were excluded from the split)")
        
        # --- START FIX 2: Check test_size vs n_classes ---
        
        # Check if we have enough samples for *any* split (both train and test need >= n_classes)
        if n_samples_strat < n_classes_strat * 2:
            tqdm.write(f"Error: Not enough samples ({n_samples_strat}) to stratify across {n_classes_strat} classes.")
            tqdm.write("  Both train and test sets must have at least one sample per class.")
            return None, None
            
        # Now, check if the *requested* test_size fraction is large enough
        final_test_size = test_size # Default (float)
        
        # Calculate number of test samples from the fraction
        default_test_n = int(n_samples_strat * test_size)
        
        if default_test_n < n_classes_strat:
            # If default test_size is too small, use the minimum possible (n_classes)
            final_test_size = n_classes_strat # Set as an INTEGER
            tqdm.write(f"⚠ Warning: Default test_size ({test_size}) gives {default_test_n} samples, which is < n_classes ({n_classes_strat}).")
            tqdm.write(f"  Using absolute test_size={final_test_size} instead.")
        
        # --- END FIX 2 ---
            
        # Split using the (potentially corrected) final_test_size
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_stratifiable,  
            labels_stratifiable,      
            test_size=final_test_size, # Use the corrected size
            stratify=labels_stratifiable, 
            random_state=42
        )
        
        tqdm.write(f"  Train set: {len(X_train)} sequences")
        tqdm.write(f"  Test set: {len(X_test)} sequences")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader, num_epochs=30, lr=1e-3):
        """Train the classifier"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training classifier", file=sys.stdout)
        
        for epoch in epoch_pbar:
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for embeddings, labels in train_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                outputs = self.model(embeddings)
                loss = criterion(outputs, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            # Testing
            test_acc = self.evaluate(test_loader)
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'train_acc': f'{train_acc:.1f}%',
                'test_acc': f'{test_acc:.1f}%'
            })
        
        epoch_pbar.close()
        tqdm.write(f"✓ Training complete - Final test accuracy: {history['test_acc'][-1]:.2f}%")
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate the classifier"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(embeddings)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def predict(self, embeddings):
        """
        Predict taxa for embeddings
        
        Args:
            embeddings: (N, D) array of embeddings
            
        Returns:
            predictions: (N,) array of predicted labels
            probabilities: (N, num_classes) array of class probabilities
        """
        self.model.eval()
        
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(embeddings_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def predict_with_confidence(self, embeddings, confidence_threshold=0.5, batch_size=128):
        """
        Predict with confidence-based unknown detection (BATCHED)
        
        Args:
            embeddings: (N, D) array of embeddings
            confidence_threshold: Minimum confidence for known classification
            batch_size: Batch size for inference (can be larger than train)
            
        Returns:
            predictions: (N,) array (-1 for unknown/low confidence)
            confidences: (N,) array of max probabilities
        """
        self.model.eval()
        
        # Create dataset and loader
        dataset = TensorDataset(torch.FloatTensor(embeddings))
        # Use a larger batch size for inference, as it doesn't need to store gradients
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
        
        all_predictions = []
        all_confidences = []
        
        tqdm.write(f"Running batched inference (batch_size={batch_size})...")
        
        with torch.no_grad():
            # Add progress bar for inference
            for (embeddings_batch,) in tqdm(loader, desc="Predicting", leave=False, file=sys.stdout): 
                embeddings_batch = embeddings_batch.to(self.device)
                
                # Get model outputs
                outputs = self.model(embeddings_batch)
                
                # Get probabilities and predictions
                probabilities = torch.softmax(outputs, dim=1)
                confidences_batch, predictions_batch = probabilities.max(dim=1)
                
                # Mark low-confidence predictions as unknown (-1)
                # We can't use -1 in a LongTensor, so we use a float tensor
                predictions_float = predictions_batch.to(torch.float32)
                predictions_float[confidences_batch < confidence_threshold] = -1
                
                all_predictions.append(predictions_float.cpu())
                all_confidences.append(confidences_batch.cpu())
        
        # Concatenate all batch results
        final_predictions = torch.cat(all_predictions).numpy().astype(int)
        final_confidences = torch.cat(all_confidences).numpy()
        
        return final_predictions, final_confidences

# ==================== Full Pipeline ====================

def analyze_cluster_taxonomy(cluster_labels, true_labels, seq_ids):
    """
    Analyze the taxonomic composition of each cluster
    
    Args:
        cluster_labels: (N,) array of cluster assignments
        true_labels: (N,) array of true taxa labels
        seq_ids: List of sequence IDs
    """
    # Group by cluster
    unique_clusters = sorted(set(cluster_labels) - {-1})
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Cluster-Taxonomy Analysis")
    tqdm.write("="*60)
    
    for cluster_id in unique_clusters[:20]:  # Show first 20 clusters
        mask = cluster_labels == cluster_id
        cluster_taxa = true_labels[mask]
        cluster_size = mask.sum()
        
        # Count taxa in cluster
        taxa_counts = Counter(cluster_taxa[cluster_taxa != -1])
        
        if len(taxa_counts) == 0:
            tqdm.write(f"\nCluster {cluster_id:3d} (size={cluster_size:4d}): All unknown")
            continue
        
        tqdm.write(f"\nCluster {cluster_id:3d} (size={cluster_size:4d}):")
        
        # Show top taxa
        for taxa, count in taxa_counts.most_common(3):
            pct = 100 * count / cluster_size
            tqdm.write(f"  Taxa {taxa:3d}: {count:4d} sequences ({pct:5.1f}%)")
        
        # Purity: fraction of most common taxa
        most_common_count = taxa_counts.most_common(1)[0][1]
        purity = most_common_count / cluster_size
        tqdm.write(f"  Purity: {purity:.3f}")
    
    if len(unique_clusters) > 20:
        tqdm.write(f"\n... and {len(unique_clusters) - 20} more clusters")

def main():
    """Main pipeline for clustering and classification"""
    
    tqdm.write("="*60)
    tqdm.write("HDBSCAN CLUSTERING + CLASSIFICATION PIPELINE")
    tqdm.write("="*60)
    
    # Configuration
    EMBEDDINGS_FILE = 'latent_embeddings.npy'
    LABELS_FILE = 'labels.npy'
    SEQ_IDS_FILE = 'seq_ids.pkl'
    
    MIN_CLUSTER_SIZE = 10  # INCREASED from 5 for faster clustering on large datasets
    MIN_SAMPLES = 5        # INCREASED from 3
    CONFIDENCE_THRESHOLD = 0.35  # For unknown detection
    
    # For very large datasets (>50k samples), you can subsample for faster clustering
    USE_SUBSAMPLING = False  # Set True if clustering is too slow
    SUBSAMPLE_SIZE = 10000   # Number of samples to use for clustering
    
    # Load embeddings
    tqdm.write("\n" + "="*60)
    tqdm.write("STEP 1: Loading Data")
    tqdm.write("="*60)
    
    embeddings = np.load(EMBEDDINGS_FILE)
    labels = np.load(LABELS_FILE)
    with open(SEQ_IDS_FILE, 'rb') as f:
        seq_ids = pickle.load(f)
    
    tqdm.write(f"✓ Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    tqdm.write(f"✓ Labeled sequences: {(labels != -1).sum()}")
    
    # Optional: Subsample for faster clustering
    if USE_SUBSAMPLING and len(embeddings) > SUBSAMPLE_SIZE:
        tqdm.write(f"\n⚠ Using subsampling: {SUBSAMPLE_SIZE} samples")
        indices = np.random.choice(len(embeddings), SUBSAMPLE_SIZE, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]
        seq_ids_subset = [seq_ids[i] for i in indices]
    else:
        embeddings_subset = embeddings
        labels_subset = labels
        seq_ids_subset = seq_ids
    
    # Step 1: Clustering with HDBSCAN
    tqdm.write("\n" + "="*60)
    tqdm.write("STEP 2: HDBSCAN Clustering")
    tqdm.write("="*60)
    
    # Suppress sklearn deprecation warnings
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    clusterer = SequenceClusterer(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES
    )
    cluster_labels = clusterer.fit(embeddings_subset)
    
    # Save cluster assignments
    np.save('cluster_labels.npy', cluster_labels)
    tqdm.write("✓ Cluster labels saved to cluster_labels.npy")
    
    # Plot cluster distribution
    clusterer.plot_cluster_distribution(cluster_labels)
    
    # Analyze cluster-taxonomy relationships
    if (labels_subset != -1).sum() > 0:
        analyze_cluster_taxonomy(cluster_labels, labels_subset, seq_ids_subset)
    
    # If we used full dataset, continue with that
    if not USE_SUBSAMPLING or len(embeddings) <= SUBSAMPLE_SIZE:
        final_embeddings = embeddings
        final_labels = labels
        final_seq_ids = seq_ids
        final_cluster_labels = cluster_labels
    else:
        tqdm.write("\n⚠ Note: Clustering was done on a subset. Full dataset not clustered.")
        final_embeddings = embeddings_subset
        final_labels = labels_subset
        final_seq_ids = seq_ids_subset
        final_cluster_labels = cluster_labels
    
    # Step 2: Train classifier (if we have labeled data)
    if (final_labels != -1).sum() > 0:
        tqdm.write("\n" + "="*60)
        tqdm.write("STEP 3: Training Taxa Classifier")
        tqdm.write("="*60)
        
        # --- START FIX 3: Remap labels to 0-based indices ---
        known_labels_mask = final_labels != -1
        unique_taxa_ids = np.unique(final_labels[known_labels_mask])
        num_classes = len(unique_taxa_ids)
        
        tqdm.write(f"Number of known taxa classes: {num_classes}")

        # Create the mapping {taxa_id -> index}
        taxa_to_index = {taxa_id: i for i, taxa_id in enumerate(unique_taxa_ids)}
        
        # Create the reverse mapping {index -> taxa_id}
        index_to_taxa = {i: taxa_id for taxa_id, i in taxa_to_index.items()}

        # Create the new indexed label array
        # Initialize all as -1 (unknown)
        indexed_labels = np.full_like(final_labels, -1)
        
        # Apply the mapping only to known labels
        # Use .get() for safety, though all known labels should be in the map
        indexed_labels[known_labels_mask] = [taxa_to_index.get(taxa_id) for taxa_id in final_labels[known_labels_mask]]
        
        tqdm.write("✓ Remapped sparse taxa IDs to 0-based indices for classifier.")
        # --- END FIX 3 ---

        classifier = TaxaClassifier(
            input_dim=final_embeddings.shape[1],
            num_classes=num_classes  # This is now the correct count
        )
        
        # Store the reverse map in the classifier for later
        classifier.index_to_taxa = index_to_taxa

        # Prepare data using the *new* indexed_labels
        train_loader, test_loader = classifier.prepare_data(final_embeddings, indexed_labels)
        
        # Handle case where prepare_data fails (e.g., not enough samples)
        if train_loader is None or test_loader is None:
            tqdm.write("Stopping: Data preparation failed.")
            sys.exit("Exiting due to data preparation error.") # Exit script
        
        # Train
        history = classifier.train(train_loader, test_loader, num_epochs=100)
        
        # Save model
        torch.save(classifier.model.state_dict(), 'taxa_classifier.pt')
        tqdm.write("✓ Classifier saved to taxa_classifier.pt")
        
        # Step 3: Predict on all data with confidence thresholding
        tqdm.write("\n" + "="*60)
        tqdm.write("STEP 4: Final Predictions")
        tqdm.write("="*60)
        
        tqdm.write("Predicting on all sequences...")
        predictions, confidences = classifier.predict_with_confidence(
            final_embeddings,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Analyze predictions
        n_known = (predictions != -1).sum()
        n_unknown = (predictions == -1).sum()
        
        tqdm.write(f"\nPrediction Summary:")
        tqdm.write(f"  Classified as known: {n_known:5d} ({100*n_known/len(predictions):.1f}%)")
        tqdm.write(f"  Classified as unknown: {n_unknown:5d} ({100*n_unknown/len(predictions):.1f}%)")
        
        # Save predictions
        np.save('final_predictions.npy', predictions)
        np.save('prediction_confidences.npy', confidences)
        tqdm.write("✓ Predictions saved to final_predictions.npy")
        tqdm.write("✓ Confidences saved to prediction_confidences.npy")
        
        # Create results file
        tqdm.write("Writing results to file...")
        with open('classification_results.txt', 'w') as f:
            f.write("seq_id\tcluster\tpredicted_taxa\tconfidence\ttrue_taxa\n")
            
            # --- START FIX 4: Convert predicted indices back to taxa IDs ---
            # Get the map from the classifier
            index_to_taxa_map = classifier.index_to_taxa 
            
            # Use tqdm for writing progress
            for i in tqdm(range(len(final_seq_ids)), desc="Writing results", leave=False, file=sys.stdout):
                pred_index = predictions[i] # This is an index (0-N) or -1
                
                # Convert predicted index back to original taxa ID
                if pred_index == -1:
                    pred_taxa = -1
                else:
                    # Look up the original taxa ID from the map
                    pred_taxa = index_to_taxa_map.get(pred_index, -999) # Use -999 to flag errors
                
                # final_labels still holds the *original* true taxa IDs
                f.write(f"{final_seq_ids[i]}\t{final_cluster_labels[i]}\t{pred_taxa}\t"
                       f"{confidences[i]:.4f}\t{final_labels[i]}\n")
            # --- END FIX 4 ---
        
        tqdm.write("✓ Results saved to classification_results.txt")
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['test_acc'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Classification Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('classifier_training.png')
        plt.close()
        tqdm.write("✓ Training curves saved to classifier_training.png")
    else:
        tqdm.write("\n⚠ No labeled data found, skipping classification training")
    
    tqdm.write("\n" + "="*60)
    tqdm.write("✓ PIPELINE COMPLETE!")
    tqdm.write("="*60)

if __name__ == '__main__':
    main()