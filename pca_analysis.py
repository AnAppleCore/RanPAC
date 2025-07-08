#!/usr/bin/env python3
"""
PCA Analysis Script for RanPAC Features_h

This script loads the saved Features_h numpy arrays from RanPAC experiments
and performs PCA analysis with visualization.

Usage:
    python pca_analysis.py --exp_path logs/model_name/dataset/init_cls/increment/exp_name
    python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/test_exp --n_components 50
    python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/test_exp --plot_only  # Fast plot generation
    python pca_analysis.py --exp_path logs/ncm/cifar224/0/10/test_exp --force_recompute  # Ignore cache
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.stats import multivariate_normal, shapiro, anderson, kstest
import pandas as pd
from typing import Dict, List, Tuple, Optional
import glob
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_features_from_experiment(exp_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load Features_h and labels from an experiment directory.
    
    Args:
        exp_path: Path to experiment directory (e.g., logs/model_name/dataset/init_cls/increment/exp_name)
        
    Returns:
        Dictionary mapping task_id -> (features, labels)
    """
    features_dir = os.path.join(exp_path, "features_h")
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Find all feature files
    feature_files = glob.glob(os.path.join(features_dir, "task_*_features_h.npy"))
    label_files = glob.glob(os.path.join(features_dir, "task_*_labels.npy"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    
    print(f"Found {len(feature_files)} feature files in {features_dir}")
    
    task_data = {}
    
    for feature_file in sorted(feature_files):
        # Extract task number from filename
        basename = os.path.basename(feature_file)
        task_id = int(basename.split('_')[1])
        
        # Find corresponding label file
        label_file = os.path.join(features_dir, f"task_{task_id}_labels.npy")
        
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for task {task_id}")
            continue
        
        # Load features and labels
        features = np.load(feature_file)
        labels = np.load(label_file)
        
        print(f"Task {task_id}: Features shape {features.shape}, Labels shape {labels.shape}")
        
        task_data[task_id] = (features, labels)
    
    return task_data

def save_pca_results(pca_results: Dict[int, Tuple[PCA, np.ndarray]], 
                    task_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                    save_path: str, 
                    n_components: int):
    """Save PCA results to disk for future use."""
    try:
        # Create the results dictionary
        results_to_save = {
            'pca_results': pca_results,
            'task_data': task_data,
            'n_components': n_components,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'num_tasks': len(task_data),
                'task_ids': list(task_data.keys()),
                'feature_shapes': {tid: features.shape for tid, (features, _) in task_data.items()}
            }
        }
        
        # Save using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"PCA results saved to: {save_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save PCA results: {e}")
        return False

def load_pca_results(save_path: str) -> Tuple[Dict[int, Tuple[PCA, np.ndarray]], 
                                             Dict[int, Tuple[np.ndarray, np.ndarray]], 
                                             dict]:
    """Load PCA results from disk."""
    try:
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        pca_results = saved_data['pca_results']
        task_data = saved_data['task_data']
        metadata = saved_data.get('metadata', {})
        
        print(f"Loaded PCA results from: {save_path}")
        print(f"Analysis timestamp: {saved_data.get('timestamp', 'Unknown')}")
        print(f"Number of tasks: {metadata.get('num_tasks', 'Unknown')}")
        
        return pca_results, task_data, metadata
    except Exception as e:
        print(f"Failed to load PCA results: {e}")
        return None, None, None

def check_pca_cache_validity(cache_path: str, 
                           task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                           n_components: int) -> bool:
    """Check if cached PCA results are still valid for current data."""
    if not os.path.exists(cache_path):
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Check if n_components matches
        if saved_data.get('n_components') != n_components:
            print(f"Cache invalid: n_components changed ({saved_data.get('n_components')} -> {n_components})")
            return False
        
        # Check if task structure matches
        saved_metadata = saved_data.get('metadata', {})
        current_task_ids = set(task_data.keys())
        saved_task_ids = set(saved_metadata.get('task_ids', []))
        
        if current_task_ids != saved_task_ids:
            print(f"Cache invalid: task structure changed")
            return False
        
        # Check if feature shapes match
        saved_shapes = saved_metadata.get('feature_shapes', {})
        for task_id, (features, _) in task_data.items():
            if task_id in saved_shapes:
                if saved_shapes[task_id] != features.shape:
                    print(f"Cache invalid: feature shape changed for task {task_id}")
                    return False
        
        print("PCA cache is valid - will load existing results")
        return True
        
    except Exception as e:
        print(f"Error checking cache validity: {e}")
        return False

def perform_pca_analysis(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                        n_components: int = 50,
                        cache_path: str = None,
                        force_recompute: bool = False) -> Dict[int, Tuple[PCA, np.ndarray]]:
    """
    Perform PCA analysis on each task's features with caching support.
    
    Args:
        task_data: Dictionary mapping task_id -> (features, labels)
        n_components: Number of PCA components to compute
        cache_path: Path to save/load PCA cache file
        force_recompute: If True, ignore cache and recompute PCA
        
    Returns:
        Dictionary mapping task_id -> (PCA object, transformed features)
    """
    # Check if we can use cached results
    if cache_path and not force_recompute and check_pca_cache_validity(cache_path, task_data, n_components):
        pca_results, cached_task_data, metadata = load_pca_results(cache_path)
        if pca_results is not None:
            return pca_results
    
    # Compute PCA (either no cache, invalid cache, or force recompute)
    print(f"Computing PCA analysis for {len(task_data)} tasks...")
    start_time = time.time()
    
    pca_results = {}
    
    for task_id, (features, labels) in task_data.items():
        print(f"\nPerforming PCA for Task {task_id}...")
        
        # Ensure we don't request more components than samples or features
        n_comp = min(n_components, features.shape[0] - 1, features.shape[1])
        
        pca = PCA(n_components=n_comp)
        features_pca = pca.fit_transform(features)
        
        print(f"Task {task_id}: Explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}")
        print(f"Task {task_id}: Cumulative explained variance (first 10 components): {np.cumsum(pca.explained_variance_ratio_[:10])}")
        
        pca_results[task_id] = (pca, features_pca)
    
    computation_time = time.time() - start_time
    print(f"\nPCA computation completed in {computation_time:.2f} seconds")
    
    # Save results to cache
    if cache_path:
        save_pca_results(pca_results, task_data, cache_path, n_components)
    
    return pca_results

def plot_explained_variance(pca_results: Dict[int, Tuple[PCA, np.ndarray]], 
                           save_dir: str, top_k: int = 20):
    """Plot explained variance for each task."""
    plt.figure(figsize=(15, 10))
    
    # Plot explained variance ratio
    plt.subplot(2, 2, 1)
    for task_id, (pca, _) in pca_results.items():
        explained_var = pca.explained_variance_ratio_[:top_k]
        plt.plot(range(1, len(explained_var) + 1), explained_var, 
                label=f'Task {task_id}', marker='o', markersize=4)
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative explained variance
    plt.subplot(2, 2, 2)
    for task_id, (pca, _) in pca_results.items():
        cumulative_var = np.cumsum(pca.explained_variance_ratio_[:top_k])
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                label=f'Task {task_id}', marker='o', markersize=4)
    
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot singular values
    plt.subplot(2, 2, 3)
    for task_id, (pca, _) in pca_results.items():
        singular_values = pca.singular_values_[:top_k]
        plt.plot(range(1, len(singular_values) + 1), singular_values, 
                label=f'Task {task_id}', marker='o', markersize=4)
    
    plt.xlabel('Principal Component')
    plt.ylabel('Singular Values')
    plt.title('Singular Values by Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot feature variance comparison
    plt.subplot(2, 2, 4)
    task_ids = list(pca_results.keys())
    first_pc_var = [pca_results[tid][0].explained_variance_ratio_[0] 
                     if len(pca_results[tid][0].explained_variance_ratio_) > 0 else 0.0 
                     for tid in task_ids]
    total_var_10 = [np.sum(pca_results[tid][0].explained_variance_ratio_[:10]) 
                    if len(pca_results[tid][0].explained_variance_ratio_) > 0 else 0.0 
                    for tid in task_ids]
    
    x = np.arange(len(task_ids))
    width = 0.35
    
    plt.bar(x - width/2, first_pc_var, width, label='1st PC', alpha=0.8)
    plt.bar(x + width/2, total_var_10, width, label='Top 10 PCs', alpha=0.8)
    
    plt.xlabel('Task ID')
    plt.ylabel('Explained Variance')
    plt.title('Variance Captured by Principal Components')
    plt.xticks(x, [f'Task {tid}' for tid in task_ids])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_2d_embeddings(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                      pca_results: Dict[int, Tuple[PCA, np.ndarray]], 
                      save_dir: str):
    """Create 2D visualizations using PCA and t-SNE."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    n_tasks = len(task_data)
    fig, axes = plt.subplots(2, n_tasks, figsize=(8 * n_tasks, 10))
    
    if n_tasks == 1:
        axes = axes.reshape(2, 1)
    
    for i, (task_id, (features, labels)) in enumerate(task_data.items()):
        pca, features_pca = pca_results[task_id]
        
        # PCA 2D plot
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = labels == label
            axes[0, i].scatter(features_pca[mask, 0], features_pca[mask, 1], 
                             c=[colors[j]], label=f'Class {label}', alpha=0.6, s=20)
        
        axes[0, i].set_title(f'Task {task_id} - PCA (2D)')
        pc1_var = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0
        pc2_var = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
        axes[0, i].set_xlabel(f'PC1 ({pc1_var:.3f})')
        axes[0, i].set_ylabel(f'PC2 ({pc2_var:.3f})')
        axes[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, i].grid(True, alpha=0.3)
        
        # t-SNE plot (using first 50 PCA components to speed up)
        n_samples = min(1000, features.shape[0])  # Limit samples for t-SNE speed
        sample_idx = np.random.choice(features.shape[0], n_samples, replace=False)
        
        features_sample = features[sample_idx]
        labels_sample = labels[sample_idx]
        
        # Use PCA features for t-SNE if available, otherwise raw features
        if features_pca.shape[1] >= 2:
            tsne_input = features_pca[sample_idx, :min(50, features_pca.shape[1])]
        else:
            tsne_input = features_sample[:, :min(50, features_sample.shape[1])]
        
        print(f"Computing t-SNE for Task {task_id} with {n_samples} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        features_tsne = tsne.fit_transform(tsne_input)
        
        for j, label in enumerate(unique_labels):
            mask = labels_sample == label
            if np.any(mask):
                axes[1, i].scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                                 c=[colors[j]], label=f'Class {label}', alpha=0.6, s=20)
        
        axes[1, i].set_title(f'Task {task_id} - t-SNE (2D)')
        axes[1, i].set_xlabel('t-SNE 1')
        axes[1, i].set_ylabel('t-SNE 2')
        axes[1, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_tsne_2d_embeddings.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_cross_task_similarity(pca_results: Dict[int, Tuple[PCA, np.ndarray]], 
                                 save_dir: str, n_components: int = 10):
    """Analyze similarity between tasks using PCA components."""
    task_ids = sorted(pca_results.keys())
    n_tasks = len(task_ids)
    
    # Extract top PCA components for each task
    task_components = {}
    for task_id in task_ids:
        pca, _ = pca_results[task_id]
        task_components[task_id] = pca.components_[:n_components]
    
    # Compute cosine similarity between PCA subspaces
    similarity_matrix = np.zeros((n_tasks, n_tasks))
    
    for i, task_i in enumerate(task_ids):
        for j, task_j in enumerate(task_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Compute similarity between subspaces
                comp_i = task_components[task_i]
                comp_j = task_components[task_j]
                
                # Use Frobenius norm of the cross-correlation matrix
                cross_corr = np.abs(comp_i @ comp_j.T)
                similarity = np.mean(np.max(cross_corr, axis=1))
                similarity_matrix[i, j] = similarity
    
    # Plot similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=[f'Task {tid}' for tid in task_ids],
                yticklabels=[f'Task {tid}' for tid in task_ids],
                annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Cross-Task PCA Subspace Similarity\n(Top {n_components} components)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_task_similarity.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return similarity_matrix

def analyze_linear_separability(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                               save_dir: str) -> Dict[int, Dict]:
    """Analyze linear separability of features for each task."""
    separability_results = {}
    
    for task_id, (features, labels) in task_data.items():
        print(f"\nAnalyzing linear separability for Task {task_id}...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get unique classes
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            print(f"Task {task_id}: Only one class, skipping separability analysis")
            continue
            
        # 1. Linear SVM with cross-validation
        try:
            svm = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
            # Ensure proper CV folds: at least 2, at most 5, and not more than samples or classes
            cv_folds = min(5, max(2, min(len(labels)//5, n_classes)))
            cv_scores = cross_val_score(svm, features_scaled, labels, cv=cv_folds, scoring='accuracy')
            svm_accuracy = cv_scores.mean()
            svm_std = cv_scores.std()
        except:
            svm_accuracy = 0.0
            svm_std = 0.0
        
        # 2. Logistic Regression with cross-validation
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            cv_folds = min(5, max(2, min(len(labels)//5, n_classes)))
            cv_scores = cross_val_score(lr, features_scaled, labels, cv=cv_folds, scoring='accuracy')
            lr_accuracy = cv_scores.mean()
            lr_std = cv_scores.std()
        except:
            lr_accuracy = 0.0
            lr_std = 0.0
        
        # 3. Linear Discriminant Analysis
        try:
            if n_classes <= features_scaled.shape[1]:  # LDA requires n_classes <= n_features
                lda = LinearDiscriminantAnalysis()
                cv_folds = min(5, max(2, min(len(labels)//5, n_classes)))
                cv_scores = cross_val_score(lda, features_scaled, labels, cv=cv_folds, scoring='accuracy')
                lda_accuracy = cv_scores.mean()
                lda_std = cv_scores.std()
            else:
                lda_accuracy = 0.0
                lda_std = 0.0
        except:
            lda_accuracy = 0.0
            lda_std = 0.0
        
        # 4. Pairwise class separability (Fisher's criterion)
        fisher_scores = []
        class_pairs = []
        
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                class_i = unique_classes[i]
                class_j = unique_classes[j]
                
                mask_i = labels == class_i
                mask_j = labels == class_j
                
                if np.sum(mask_i) > 1 and np.sum(mask_j) > 1:
                    features_i = features_scaled[mask_i]
                    features_j = features_scaled[mask_j]
                    
                    # Compute Fisher's criterion
                    mean_i = np.mean(features_i, axis=0)
                    mean_j = np.mean(features_j, axis=0)
                    var_i = np.var(features_i, axis=0)
                    var_j = np.var(features_j, axis=0)
                    
                    # Fisher's criterion: (mean_diff)^2 / (var_i + var_j)
                    fisher_criterion = np.mean((mean_i - mean_j)**2 / (var_i + var_j + 1e-10))
                    fisher_scores.append(fisher_criterion)
                    class_pairs.append((class_i, class_j))
        
        mean_fisher = np.mean(fisher_scores) if fisher_scores else 0.0
        
        # 5. Class centroid distances relative to within-class variance
        class_centroids = []
        within_class_vars = []
        
        for class_id in unique_classes:
            mask = labels == class_id
            if np.sum(mask) > 1:
                class_features = features_scaled[mask]
                centroid = np.mean(class_features, axis=0)
                within_var = np.mean(np.var(class_features, axis=0))
                class_centroids.append(centroid)
                within_class_vars.append(within_var)
        
        if len(class_centroids) > 1:
            # Compute pairwise distances between centroids
            centroid_distances = []
            for i in range(len(class_centroids)):
                for j in range(i+1, len(class_centroids)):
                    dist = np.linalg.norm(class_centroids[i] - class_centroids[j])
                    centroid_distances.append(dist)
            
            mean_centroid_distance = np.mean(centroid_distances)
            mean_within_var = np.mean(within_class_vars)
            separability_ratio = mean_centroid_distance / (np.sqrt(mean_within_var) + 1e-10)
        else:
            separability_ratio = 0.0
        
        # Store results
        separability_results[task_id] = {
            'n_classes': n_classes,
            'n_samples': len(labels),
            'n_features': features_scaled.shape[1],
            'svm_accuracy': svm_accuracy,
            'svm_std': svm_std,
            'lr_accuracy': lr_accuracy,
            'lr_std': lr_std,
            'lda_accuracy': lda_accuracy,
            'lda_std': lda_std,
            'fisher_score': mean_fisher,
            'separability_ratio': separability_ratio,
            'class_pairs_fisher': list(zip(class_pairs, fisher_scores)) if fisher_scores else []
        }
        
        print(f"Task {task_id} Linear Separability:")
        print(f"  SVM Accuracy: {svm_accuracy:.3f} ± {svm_std:.3f}")
        print(f"  LR Accuracy: {lr_accuracy:.3f} ± {lr_std:.3f}")
        print(f"  LDA Accuracy: {lda_accuracy:.3f} ± {lda_std:.3f}")
        print(f"  Fisher Score: {mean_fisher:.3f}")
        print(f"  Separability Ratio: {separability_ratio:.3f}")
    
    return separability_results

def analyze_gaussian_assumption(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                               save_dir: str, n_test_features: int = 10) -> Dict[int, Dict]:
    """Analyze how well features follow Gaussian distributions for each class."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    gaussian_results = {}
    
    for task_id, (features, labels) in task_data.items():
        print(f"\nAnalyzing Gaussian assumption for Task {task_id}...")
        
        unique_classes = np.unique(labels)
        task_results = {}
        
        for class_id in unique_classes:
            mask = labels == class_id
            class_features = features[mask]
            
            if len(class_features) < 5:  # Need at least 5 samples for meaningful tests
                continue
                
            class_results = {
                'n_samples': len(class_features),
                'feature_tests': {},
                'multivariate_test': {}
            }
            
            # Test subset of features for normality (to avoid too many tests)
            n_features_to_test = min(n_test_features, class_features.shape[1])
            feature_indices = np.random.choice(class_features.shape[1], n_features_to_test, replace=False)
            
            shapiro_pvalues = []
            anderson_statistics = []
            ks_pvalues = []
            
            for feat_idx in feature_indices:
                feature_values = class_features[:, feat_idx]
                
                # Shapiro-Wilk test (best for small samples)
                if len(feature_values) <= 5000:  # Shapiro-Wilk limitation
                    try:
                        _, p_shapiro = shapiro(feature_values)
                        shapiro_pvalues.append(p_shapiro)
                    except:
                        shapiro_pvalues.append(0.0)
                else:
                    shapiro_pvalues.append(0.0)
                
                # Anderson-Darling test
                try:
                    ad_stat, _, _ = anderson(feature_values, dist='norm')
                    anderson_statistics.append(ad_stat)
                except:
                    anderson_statistics.append(np.inf)
                
                # Kolmogorov-Smirnov test against normal distribution
                try:
                    # Standardize first
                    standardized = (feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-10)
                    _, p_ks = kstest(standardized, 'norm')
                    ks_pvalues.append(p_ks)
                except:
                    ks_pvalues.append(0.0)
            
            class_results['feature_tests'] = {
                'shapiro_pvalues': shapiro_pvalues,
                'shapiro_mean': np.mean(shapiro_pvalues),
                'shapiro_fraction_normal': np.mean(np.array(shapiro_pvalues) > 0.05),
                'anderson_statistics': anderson_statistics,
                'anderson_mean': np.mean(anderson_statistics),
                'ks_pvalues': ks_pvalues,
                'ks_mean': np.mean(ks_pvalues),
                'ks_fraction_normal': np.mean(np.array(ks_pvalues) > 0.05),
                'tested_features': n_features_to_test
            }
            
            # Multivariate normality test (simplified)
            if class_features.shape[1] <= 50 and len(class_features) > class_features.shape[1]:
                try:
                    # Use a subset of features for multivariate test
                    max_features = min(10, class_features.shape[1])
                    subset_features = class_features[:, :max_features]
                    
                    # Compute Mahalanobis distances
                    mean_vec = np.mean(subset_features, axis=0)
                    cov_matrix = np.cov(subset_features.T)
                    
                    # Add regularization to avoid singular matrix
                    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                    
                    try:
                        inv_cov = np.linalg.inv(cov_matrix)
                        mahal_distances = []
                        for sample in subset_features:
                            diff = sample - mean_vec
                            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                            mahal_distances.append(mahal_dist)
                        
                        # Chi-square test for multivariate normality
                        # Mahalanobis distances should follow chi-square distribution
                        mahal_distances = np.array(mahal_distances)
                        _, p_multivar = kstest(mahal_distances, lambda x: stats.chi2.cdf(x, df=max_features))
                        
                        class_results['multivariate_test'] = {
                            'mahalanobis_ks_pvalue': p_multivar,
                            'features_tested': max_features,
                            'is_multivariate_normal': p_multivar > 0.05
                        }
                    except:
                        class_results['multivariate_test'] = {'error': 'Singular covariance matrix'}
                except:
                    class_results['multivariate_test'] = {'error': 'Multivariate test failed'}
            else:
                class_results['multivariate_test'] = {'error': 'Too many features or too few samples'}
            
            task_results[class_id] = class_results
            
            # Print summary for this class
            shapiro_frac = class_results['feature_tests']['shapiro_fraction_normal']
            ks_frac = class_results['feature_tests']['ks_fraction_normal']
            print(f"  Class {class_id}: {shapiro_frac:.1%} features pass Shapiro-Wilk, {ks_frac:.1%} pass KS test")
        
        gaussian_results[task_id] = task_results
    
    return gaussian_results

def analyze_combined_features(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                             save_dir: str) -> Tuple[Dict, Dict]:
    """Analyze linear separability and Gaussian assumptions for combined features from all tasks."""
    print("\nAnalyzing combined features from all tasks...")
    
    # Check if we have any data
    if not task_data:
        print("No task data available for combined analysis")
        return {}, {}
    
    # Combine all features and labels
    all_features = []
    all_labels = []
    task_ids = []
    
    for task_id, (features, labels) in task_data.items():
        if features.size == 0 or labels.size == 0:
            print(f"Warning: Task {task_id} has empty features or labels, skipping")
            continue
        all_features.append(features)
        all_labels.append(labels)
        task_ids.extend([task_id] * len(labels))
    
    if not all_features:
        print("No valid features found for combined analysis")
        return {}, {}
    
    # Check if we have enough data for meaningful analysis
    if len(all_features) == 1:
        print("Warning: Only one task with data, combined analysis may be limited")
    
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    task_ids = np.array(task_ids)
    
    print(f"Combined dataset: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
    print(f"Number of unique classes: {len(np.unique(combined_labels))}")
    print(f"Number of tasks: {len(np.unique(task_ids))}")
    
    # Validate that we have sufficient data
    if combined_features.shape[0] < 10:
        print("Warning: Very few samples for combined analysis, results may be unreliable")
    
    if len(np.unique(combined_labels)) < 2:
        print("Warning: Less than 2 classes in combined dataset, separability analysis will be limited")
    
    # Create a combined task_data dictionary for compatibility with existing functions
    combined_task_data = {-1: (combined_features, combined_labels)}  # Use -1 to indicate combined
    
    # Analyze linear separability for combined features
    print("Performing linear separability analysis on combined features...")
    try:
        combined_separability = analyze_linear_separability(combined_task_data, save_dir)
        if not combined_separability:
            print("Warning: Linear separability analysis returned no results")
            combined_separability = {}
    except Exception as e:
        print(f"Linear separability analysis failed: {e}")
        combined_separability = {}
    
    # Analyze Gaussian assumptions for combined features
    print("Performing Gaussian assumption analysis on combined features...")
    try:
        combined_gaussian = analyze_gaussian_assumption(combined_task_data, save_dir, n_test_features=20)
        if not combined_gaussian:
            print("Warning: Gaussian assumption analysis returned no results")
            combined_gaussian = {}
    except Exception as e:
        print(f"Gaussian assumption analysis failed: {e}")
        combined_gaussian = {}
    
    # Add additional analysis specific to combined features
    print("Performing additional cross-task analysis...")
    
    # Task-wise separability: can we distinguish which task each sample came from?
    task_separability = {}
    if len(np.unique(task_ids)) > 1:
        try:
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(combined_features)
            
            # Test if we can predict task ID from features
            svm_task = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
            cv_scores = cross_val_score(svm_task, features_scaled, task_ids, cv=5, scoring='accuracy')
            task_separability = {
                'task_prediction_accuracy': cv_scores.mean(),
                'task_prediction_std': cv_scores.std(),
                'n_tasks': len(np.unique(task_ids))
            }
            print(f"Task prediction accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            print(f"Task separability analysis failed: {e}")
            task_separability = {'error': str(e)}
    
    # Inter-task class confusion: how well can we separate classes across different tasks?
    inter_task_analysis = {}
    try:
        unique_classes = np.unique(combined_labels)
        unique_tasks = np.unique(task_ids)
        
        # For each class, check if samples from different tasks are separable
        class_task_separability = {}
        
        for class_id in unique_classes:
            class_mask = combined_labels == class_id
            class_features = combined_features[class_mask]
            class_task_ids = task_ids[class_mask]
            
            unique_class_tasks = np.unique(class_task_ids)
            if len(unique_class_tasks) > 1:  # Class appears in multiple tasks
                try:
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(class_features)
                    
                    svm = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
                    cv_scores = cross_val_score(svm, features_scaled, class_task_ids, cv=3, scoring='accuracy')
                    
                    class_task_separability[class_id] = {
                        'accuracy': cv_scores.mean(),
                        'std': cv_scores.std(),
                        'n_tasks_with_class': len(unique_class_tasks),
                        'n_samples': len(class_features)
                    }
                except:
                    class_task_separability[class_id] = {'error': 'Analysis failed'}
        
        inter_task_analysis = {
            'class_task_separability': class_task_separability,
            'classes_in_multiple_tasks': len(class_task_separability)
        }
        
    except Exception as e:
        print(f"Inter-task analysis failed: {e}")
        inter_task_analysis = {'error': str(e)}
    
    # Combine all additional analyses
    if combined_separability and -1 in combined_separability:
        combined_separability[-1]['task_separability'] = task_separability
        combined_separability[-1]['inter_task_analysis'] = inter_task_analysis
    elif combined_separability is not None:
        # If combined_separability exists but doesn't have -1 key, create it
        combined_separability[-1] = {
            'task_separability': task_separability,
            'inter_task_analysis': inter_task_analysis
        }
    
    return combined_separability, combined_gaussian

def plot_combined_analysis(combined_separability: Dict, combined_gaussian: Dict, 
                          task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], save_dir: str):
    """Create visualization for combined feature analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if -1 not in combined_separability or -1 not in combined_gaussian:
        return
    
    sep_result = combined_separability[-1]
    gauss_result = combined_gaussian[-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Combined vs Individual Task Separability Comparison
    individual_svm_accs = []
    individual_fisher_scores = []
    task_ids = sorted([tid for tid in task_data.keys()])
    
    # We'll need to compute individual task separability for comparison
    for task_id in task_ids:
        task_features, task_labels = task_data[task_id]
        if len(np.unique(task_labels)) > 1:
            # Quick SVM accuracy computation
            try:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(task_features)
                svm = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
                n_classes_task = len(np.unique(task_labels))
                cv_folds = min(5, max(2, min(len(task_labels)//5, n_classes_task)))
                cv_scores = cross_val_score(svm, features_scaled, task_labels, cv=cv_folds, scoring='accuracy')
                individual_svm_accs.append(cv_scores.mean())
                
                # Quick Fisher score
                unique_classes = np.unique(task_labels)
                fisher_scores = []
                for i in range(len(unique_classes)):
                    for j in range(i+1, len(unique_classes)):
                        mask_i = task_labels == unique_classes[i]
                        mask_j = task_labels == unique_classes[j]
                        if np.sum(mask_i) > 1 and np.sum(mask_j) > 1:
                            mean_i = np.mean(features_scaled[mask_i], axis=0)
                            mean_j = np.mean(features_scaled[mask_j], axis=0)
                            var_i = np.var(features_scaled[mask_i], axis=0)
                            var_j = np.var(features_scaled[mask_j], axis=0)
                            fisher_criterion = np.mean((mean_i - mean_j)**2 / (var_i + var_j + 1e-10))
                            fisher_scores.append(fisher_criterion)
                individual_fisher_scores.append(np.mean(fisher_scores) if fisher_scores else 0)
            except:
                individual_svm_accs.append(0)
                individual_fisher_scores.append(0)
        else:
            individual_svm_accs.append(0)
            individual_fisher_scores.append(0)
    
    # Bar plot comparing individual vs combined
    x_pos = np.arange(len(task_ids) + 1)
    all_accs = individual_svm_accs + [sep_result['svm_accuracy']]
    colors = ['skyblue'] * len(task_ids) + ['red']
    labels = [f'Task {tid}' for tid in task_ids] + ['Combined']
    
    bars = axes[0, 0].bar(x_pos, all_accs, color=colors, alpha=0.7)
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('Linear SVM Accuracy')
    axes[0, 0].set_title('Individual vs Combined Feature Separability')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight the combined result
    bars[-1].set_edgecolor('darkred')
    bars[-1].set_linewidth(2)
    
    # Plot 2: Task Prediction Analysis
    if 'task_separability' in sep_result and 'task_prediction_accuracy' in sep_result['task_separability']:
        task_pred_acc = sep_result['task_separability']['task_prediction_accuracy']
        n_tasks = sep_result['task_separability']['n_tasks']
        random_baseline = 1.0 / n_tasks
        
        axes[0, 1].bar(['Random\nBaseline', 'Feature-based\nPrediction'], 
                       [random_baseline, task_pred_acc], 
                       color=['gray', 'orange'], alpha=0.7)
        axes[0, 1].set_ylabel('Task Prediction Accuracy')
        axes[0, 1].set_title('Task Separability Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add text annotation
        y_offset = min(0.05, (1.0 - task_pred_acc) * 0.1)  # Ensure text stays within bounds
        axes[0, 1].text(1, task_pred_acc + y_offset, f'{task_pred_acc:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Task Separability\nAnalysis Failed', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Task Separability Analysis')
    
    # Plot 3: Combined Feature Distribution Quality
    metrics = ['SVM Accuracy', 'LR Accuracy', 'LDA Accuracy']
    values = [sep_result['svm_accuracy'], sep_result['lr_accuracy'], sep_result['lda_accuracy']]
    
    bars = axes[0, 2].bar(metrics, values, color=['blue', 'green', 'purple'], alpha=0.7)
    axes[0, 2].set_ylabel('Classification Accuracy')
    axes[0, 2].set_title('Combined Features: Linear Classifier Performance')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Gaussian Assumption - Combined vs Individual
    individual_shapiro_fractions = []
    for task_id in task_ids:
        # We need to compute this quickly for comparison
        task_features, task_labels = task_data[task_id]
        unique_classes = np.unique(task_labels)
        task_shapiro_fractions = []
        
        for class_id in unique_classes:
            mask = task_labels == class_id
            class_features = task_features[mask]
            if len(class_features) >= 5:
                # Test a few features
                n_test = min(5, class_features.shape[1])
                feature_indices = np.random.choice(class_features.shape[1], n_test, replace=False)
                shapiro_pvalues = []
                
                for feat_idx in feature_indices:
                    feature_values = class_features[:, feat_idx]
                    try:
                        if len(feature_values) <= 5000:
                            _, p_shapiro = stats.shapiro(feature_values)
                            shapiro_pvalues.append(p_shapiro)
                    except:
                        pass
                
                if shapiro_pvalues:
                    task_shapiro_fractions.append(np.mean(np.array(shapiro_pvalues) > 0.05))
        
        if task_shapiro_fractions:
            individual_shapiro_fractions.append(np.mean(task_shapiro_fractions))
        else:
            individual_shapiro_fractions.append(0)
    
    # Get combined Gaussian results
    combined_shapiro_fractions = []
    for class_id, class_result in gauss_result.items():
        if 'feature_tests' in class_result:
            combined_shapiro_fractions.append(class_result['feature_tests']['shapiro_fraction_normal'])
    
    combined_avg = np.mean(combined_shapiro_fractions) if combined_shapiro_fractions else 0
    
    # Plot comparison
    x_pos = np.arange(len(task_ids) + 1)
    all_gaussian = individual_shapiro_fractions + [combined_avg]
    colors = ['lightgreen'] * len(task_ids) + ['darkgreen']
    
    bars = axes[1, 0].bar(x_pos, all_gaussian, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('Fraction Passing Shapiro-Wilk')
    axes[1, 0].set_title('Gaussian Assumption: Individual vs Combined')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 5: Inter-task Class Analysis
    if 'inter_task_analysis' in sep_result and 'class_task_separability' in sep_result['inter_task_analysis']:
        class_task_sep = sep_result['inter_task_analysis']['class_task_separability']
        
        if class_task_sep:
            class_ids = []
            accuracies = []
            n_tasks_per_class = []
            
            for class_id, result in class_task_sep.items():
                if 'accuracy' in result:
                    class_ids.append(class_id)
                    accuracies.append(result['accuracy'])
                    n_tasks_per_class.append(result['n_tasks_with_class'])
            
            if class_ids:
                scatter = axes[1, 1].scatter(n_tasks_per_class, accuracies, 
                                           c=class_ids, s=100, alpha=0.6, cmap='tab10')
                axes[1, 1].set_xlabel('Number of Tasks Containing Class')
                axes[1, 1].set_ylabel('Task Prediction Accuracy for Class')
                axes[1, 1].set_title('Per-Class Task Separability')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[1, 1])
                cbar.set_label('Class ID')
            else:
                axes[1, 1].text(0.5, 0.5, 'No classes appear\nin multiple tasks', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Per-Class Task Separability')
        else:
            axes[1, 1].text(0.5, 0.5, 'Inter-task Analysis\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Per-Class Task Separability')
    else:
        axes[1, 1].text(0.5, 0.5, 'Inter-task Analysis\nFailed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Per-Class Task Separability')
    
    # Plot 6: Overall Quality Summary
    # Create a radar-like summary of all metrics
    metrics = ['Linear\nSeparability', 'Gaussian\nAssumption', 'Fisher\nScore', 'Task\nSeparability']
    
    # Normalize metrics to 0-1 scale
    linear_sep = sep_result['svm_accuracy']
    gaussian_qual = combined_avg
    fisher_norm = min(1.0, sep_result['fisher_score'] / 10.0)  # Normalize Fisher score
    task_sep = sep_result.get('task_separability', {}).get('task_prediction_accuracy', 0)
    
    values = [linear_sep, gaussian_qual, fisher_norm, task_sep]
    
    bars = axes[1, 2].bar(metrics, values, 
                         color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1, 2].set_ylabel('Quality Score (0-1)')
    axes[1, 2].set_title('Combined Features: Overall Quality Summary')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_features_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_separability_analysis(separability_results: Dict[int, Dict], save_dir: str):
    """Plot linear separability analysis results."""
    if not separability_results:
        return
    
    task_ids = sorted(separability_results.keys())
    
    # Extract metrics for plotting
    svm_accs = [separability_results[tid]['svm_accuracy'] for tid in task_ids]
    svm_stds = [separability_results[tid]['svm_std'] for tid in task_ids]
    lr_accs = [separability_results[tid]['lr_accuracy'] for tid in task_ids]
    lr_stds = [separability_results[tid]['lr_std'] for tid in task_ids]
    lda_accs = [separability_results[tid]['lda_accuracy'] for tid in task_ids]
    fisher_scores = [separability_results[tid]['fisher_score'] for tid in task_ids]
    sep_ratios = [separability_results[tid]['separability_ratio'] for tid in task_ids]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Classification accuracies
    x = np.arange(len(task_ids))
    width = 0.25
    
    axes[0, 0].bar(x - width, svm_accs, width, yerr=svm_stds, label='Linear SVM', alpha=0.8, capsize=5)
    axes[0, 0].bar(x, lr_accs, width, yerr=lr_stds, label='Logistic Regression', alpha=0.8, capsize=5)
    axes[0, 0].bar(x + width, lda_accs, width, label='LDA', alpha=0.8, capsize=5)
    
    axes[0, 0].set_xlabel('Task ID')
    axes[0, 0].set_ylabel('Classification Accuracy')
    axes[0, 0].set_title('Linear Classifier Performance (Cross-Validation)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Task {tid}' for tid in task_ids])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Fisher scores
    axes[0, 1].plot(task_ids, fisher_scores, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Task ID')
    axes[0, 1].set_ylabel('Fisher Score')
    axes[0, 1].set_title('Fisher\'s Linear Discriminant Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Separability ratios
    axes[1, 0].plot(task_ids, sep_ratios, 's-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Task ID')
    axes[1, 0].set_ylabel('Separability Ratio')
    axes[1, 0].set_title('Class Centroid Distance / Within-Class Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined metric comparison
    # Normalize metrics to [0, 1] for comparison
    norm_svm = np.array(svm_accs)
    
    # Safe normalization for Fisher scores
    fisher_max = np.max(fisher_scores) if fisher_scores and np.max(fisher_scores) > 0 else 1.0
    norm_fisher = np.array(fisher_scores) / fisher_max
    
    # Safe normalization for separability ratios
    sep_max = np.max(sep_ratios) if sep_ratios and np.max(sep_ratios) > 0 else 1.0
    norm_sep = np.array(sep_ratios) / sep_max
    
    axes[1, 1].plot(task_ids, norm_svm, 'o-', label='SVM Accuracy', linewidth=2)
    axes[1, 1].plot(task_ids, norm_fisher, 's-', label='Fisher Score (norm)', linewidth=2)
    axes[1, 1].plot(task_ids, norm_sep, '^-', label='Sep. Ratio (norm)', linewidth=2)
    
    axes[1, 1].set_xlabel('Task ID')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].set_title('Linear Separability Metrics Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'linear_separability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_gaussian_analysis(gaussian_results: Dict[int, Dict], save_dir: str):
    """Plot Gaussian distribution analysis results."""
    if not gaussian_results:
        return
    
    # Collect data for plotting
    task_data = []
    
    for task_id, task_result in gaussian_results.items():
        for class_id, class_result in task_result.items():
            if 'feature_tests' in class_result:
                task_data.append({
                    'task_id': task_id,
                    'class_id': class_id,
                    'n_samples': class_result['n_samples'],
                    'shapiro_fraction': class_result['feature_tests']['shapiro_fraction_normal'],
                    'ks_fraction': class_result['feature_tests']['ks_fraction_normal'],
                    'shapiro_mean_p': class_result['feature_tests']['shapiro_mean'],
                    'ks_mean_p': class_result['feature_tests']['ks_mean']
                })
    
    if not task_data:
        return
    
    df = pd.DataFrame(task_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Fraction of features passing normality tests by task
    task_summary = df.groupby('task_id').agg({
        'shapiro_fraction': 'mean',
        'ks_fraction': 'mean'
    }).reset_index()
    
    x = np.arange(len(task_summary))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, task_summary['shapiro_fraction'], width, 
                   label='Shapiro-Wilk', alpha=0.8)
    axes[0, 0].bar(x + width/2, task_summary['ks_fraction'], width, 
                   label='Kolmogorov-Smirnov', alpha=0.8)
    
    axes[0, 0].set_xlabel('Task ID')
    axes[0, 0].set_ylabel('Fraction of Features Passing Test')
    axes[0, 0].set_title('Gaussian Assumption: Feature-wise Tests by Task')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Task {tid}' for tid in task_summary['task_id']])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Distribution of p-values across all classes
    all_shapiro_p = df['shapiro_mean_p'].values
    all_ks_p = df['ks_mean_p'].values
    
    axes[0, 1].hist(all_shapiro_p, bins=20, alpha=0.7, label='Shapiro-Wilk p-values', density=True)
    axes[0, 1].hist(all_ks_p, bins=20, alpha=0.7, label='KS p-values', density=True)
    axes[0, 1].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    axes[0, 1].set_xlabel('p-value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Normality Test p-values')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of normality by task and class
    pivot_shapiro = df.pivot(index='task_id', columns='class_id', values='shapiro_fraction')
    if not pivot_shapiro.empty:
        sns.heatmap(pivot_shapiro, annot=True, fmt='.2f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=axes[1, 0])
        axes[1, 0].set_title('Gaussian Assumption: Shapiro-Wilk Fraction by Task & Class')
        axes[1, 0].set_xlabel('Class ID')
        axes[1, 0].set_ylabel('Task ID')
    
    # Plot 4: Sample size vs normality
    axes[1, 1].scatter(df['n_samples'], df['shapiro_fraction'], alpha=0.6, 
                      label='Shapiro-Wilk', s=50)
    axes[1, 1].scatter(df['n_samples'], df['ks_fraction'], alpha=0.6, 
                      label='KS Test', s=50)
    axes[1, 1].set_xlabel('Number of Samples per Class')
    axes[1, 1].set_ylabel('Fraction of Features Passing Test')
    axes[1, 1].set_title('Sample Size vs Normality Test Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gaussian_assumption_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_summary(task_data: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                         pca_results: Dict[int, Tuple[PCA, np.ndarray]], 
                         similarity_matrix: np.ndarray,
                         separability_results: Dict[int, Dict],
                         gaussian_results: Dict[int, Dict],
                         save_dir: str,
                         combined_separability: Dict = None,
                         combined_gaussian: Dict = None):
    """Save a comprehensive summary of all analyses."""
    summary = []
    task_ids = sorted(task_data.keys())
    
    summary.append("=" * 80)
    summary.append("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # PCA Analysis Summary
    summary.append("PCA ANALYSIS:")
    summary.append("-" * 40)
    for task_id in task_ids:
        features, labels = task_data[task_id]
        pca, features_pca = pca_results[task_id]
        
        summary.append(f"TASK {task_id}:")
        summary.append(f"  Original features shape: {features.shape}")
        summary.append(f"  Number of classes: {len(np.unique(labels))}")
        summary.append(f"  Classes: {sorted(np.unique(labels))}")
        summary.append(f"  PCA components: {features_pca.shape[1]}")
        if len(pca.explained_variance_ratio_) > 0:
            summary.append(f"  First PC explains: {pca.explained_variance_ratio_[0]:.4f} of variance")
            summary.append(f"  Top 5 PCs explain: {np.sum(pca.explained_variance_ratio_[:5]):.4f} of variance")
            summary.append(f"  Top 10 PCs explain: {np.sum(pca.explained_variance_ratio_[:10]):.4f} of variance")
        summary.append("")
    
    # Cross-task Similarity
    summary.append("CROSS-TASK SIMILARITY:")
    summary.append("-" * 40)
    for i, task_i in enumerate(task_ids):
        for j, task_j in enumerate(task_ids):
            if i < j:
                similarity = similarity_matrix[i, j]
                summary.append(f"  Task {task_i} vs Task {task_j}: {similarity:.4f}")
    summary.append("")
    
    # Linear Separability Summary
    if separability_results:
        summary.append("LINEAR SEPARABILITY ANALYSIS:")
        summary.append("-" * 40)
        for task_id in task_ids:
            if task_id in separability_results:
                sep_result = separability_results[task_id]
                summary.append(f"TASK {task_id}:")
                summary.append(f"  Classes: {sep_result['n_classes']}, Samples: {sep_result['n_samples']}, Features: {sep_result['n_features']}")
                summary.append(f"  Linear SVM Accuracy: {sep_result['svm_accuracy']:.3f} ± {sep_result['svm_std']:.3f}")
                summary.append(f"  Logistic Regression Accuracy: {sep_result['lr_accuracy']:.3f} ± {sep_result['lr_std']:.3f}")
                summary.append(f"  LDA Accuracy: {sep_result['lda_accuracy']:.3f} ± {sep_result['lda_std']:.3f}")
                summary.append(f"  Fisher Score (higher = more separable): {sep_result['fisher_score']:.3f}")
                summary.append(f"  Separability Ratio (higher = more separable): {sep_result['separability_ratio']:.3f}")
                
                # Interpretation
                best_acc = max(sep_result['svm_accuracy'], sep_result['lr_accuracy'], sep_result['lda_accuracy'])
                if best_acc > 0.9:
                    separability_level = "Excellent linear separability"
                elif best_acc > 0.8:
                    separability_level = "Good linear separability"
                elif best_acc > 0.7:
                    separability_level = "Moderate linear separability"
                elif best_acc > 0.6:
                    separability_level = "Poor linear separability"
                else:
                    separability_level = "Very poor linear separability"
                
                summary.append(f"  → {separability_level}")
                summary.append("")
        
        # Overall separability trends
        all_svm_accs = [separability_results[tid]['svm_accuracy'] for tid in task_ids if tid in separability_results]
        if all_svm_accs:
            summary.append("SEPARABILITY TRENDS:")
            summary.append(f"  Mean Linear SVM Accuracy across tasks: {np.mean(all_svm_accs):.3f}")
            summary.append(f"  Std Linear SVM Accuracy across tasks: {np.std(all_svm_accs):.3f}")
            summary.append(f"  Min/Max SVM Accuracy: {np.min(all_svm_accs):.3f} / {np.max(all_svm_accs):.3f}")
            summary.append("")
    
    # Gaussian Assumption Summary
    if gaussian_results:
        summary.append("GAUSSIAN DISTRIBUTION ANALYSIS:")
        summary.append("-" * 40)
        
        all_shapiro_fractions = []
        all_ks_fractions = []
        
        for task_id in task_ids:
            if task_id in gaussian_results:
                summary.append(f"TASK {task_id}:")
                task_result = gaussian_results[task_id]
                
                task_shapiro_fractions = []
                task_ks_fractions = []
                
                for class_id, class_result in task_result.items():
                    if 'feature_tests' in class_result:
                        shapiro_frac = class_result['feature_tests']['shapiro_fraction_normal']
                        ks_frac = class_result['feature_tests']['ks_fraction_normal']
                        n_samples = class_result['n_samples']
                        
                        task_shapiro_fractions.append(shapiro_frac)
                        task_ks_fractions.append(ks_frac)
                        
                        summary.append(f"  Class {class_id} ({n_samples} samples):")
                        summary.append(f"    Shapiro-Wilk: {shapiro_frac:.1%} of features pass normality test")
                        summary.append(f"    KS Test: {ks_frac:.1%} of features pass normality test")
                        
                        # Multivariate normality
                        if 'multivariate_test' in class_result and 'mahalanobis_ks_pvalue' in class_result['multivariate_test']:
                            mv_pvalue = class_result['multivariate_test']['mahalanobis_ks_pvalue']
                            mv_normal = class_result['multivariate_test']['is_multivariate_normal']
                            summary.append(f"    Multivariate normality: {'Yes' if mv_normal else 'No'} (p={mv_pvalue:.3f})")
                
                if task_shapiro_fractions:
                    mean_shapiro = np.mean(task_shapiro_fractions)
                    mean_ks = np.mean(task_ks_fractions)
                    summary.append(f"  Task {task_id} Average: Shapiro {mean_shapiro:.1%}, KS {mean_ks:.1%}")
                    
                    all_shapiro_fractions.extend(task_shapiro_fractions)
                    all_ks_fractions.extend(task_ks_fractions)
                    
                    # Interpretation
                    if mean_shapiro > 0.7:
                        gaussian_level = "Strong Gaussian assumption"
                    elif mean_shapiro > 0.5:
                        gaussian_level = "Moderate Gaussian assumption"
                    elif mean_shapiro > 0.3:
                        gaussian_level = "Weak Gaussian assumption"
                    else:
                        gaussian_level = "Poor Gaussian assumption"
                    
                    summary.append(f"  → {gaussian_level}")
                summary.append("")
        
        # Overall Gaussian trends
        if all_shapiro_fractions:
            summary.append("GAUSSIAN ASSUMPTION TRENDS:")
            summary.append(f"  Overall Shapiro-Wilk pass rate: {np.mean(all_shapiro_fractions):.1%}")
            summary.append(f"  Overall KS test pass rate: {np.mean(all_ks_fractions):.1%}")
            summary.append(f"  Standard deviation across classes: Shapiro {np.std(all_shapiro_fractions):.3f}, KS {np.std(all_ks_fractions):.3f}")
            summary.append("")
    
    # Combined Features Analysis Summary
    if combined_separability and -1 in combined_separability:
        summary.append("COMBINED FEATURES ANALYSIS:")
        summary.append("-" * 40)
        combined_sep = combined_separability[-1]
        
        # Get total samples and features
        total_samples = sum(task_data[tid][0].shape[0] for tid in task_ids)
        total_features = task_data[task_ids[0]][0].shape[1] if task_ids else 0
        total_classes = len(set().union(*[np.unique(task_data[tid][1]) for tid in task_ids]))
        
        summary.append(f"COMBINED DATASET:")
        summary.append(f"  Total samples: {total_samples}")
        summary.append(f"  Feature dimensions: {total_features}")
        summary.append(f"  Total unique classes: {total_classes}")
        summary.append(f"  Number of tasks: {len(task_ids)}")
        summary.append("")
        
        summary.append("COMBINED LINEAR SEPARABILITY:")
        summary.append(f"  Linear SVM Accuracy: {combined_sep['svm_accuracy']:.3f} ± {combined_sep['svm_std']:.3f}")
        summary.append(f"  Logistic Regression Accuracy: {combined_sep['lr_accuracy']:.3f} ± {combined_sep['lr_std']:.3f}")
        summary.append(f"  LDA Accuracy: {combined_sep['lda_accuracy']:.3f} ± {combined_sep['lda_std']:.3f}")
        summary.append(f"  Fisher Score: {combined_sep['fisher_score']:.3f}")
        summary.append(f"  Separability Ratio: {combined_sep['separability_ratio']:.3f}")
        
        # Interpretation for combined features
        best_combined_acc = max(combined_sep['svm_accuracy'], combined_sep['lr_accuracy'], combined_sep['lda_accuracy'])
        if best_combined_acc > 0.9:
            combined_sep_level = "Excellent combined linear separability"
        elif best_combined_acc > 0.8:
            combined_sep_level = "Good combined linear separability"
        elif best_combined_acc > 0.7:
            combined_sep_level = "Moderate combined linear separability"
        elif best_combined_acc > 0.6:
            combined_sep_level = "Poor combined linear separability"
        else:
            combined_sep_level = "Very poor combined linear separability"
        
        summary.append(f"  → {combined_sep_level}")
        summary.append("")
        
        # Task separability analysis
        if 'task_separability' in combined_sep and 'task_prediction_accuracy' in combined_sep['task_separability']:
            task_sep = combined_sep['task_separability']
            random_baseline = 1.0 / task_sep['n_tasks']
            summary.append("TASK SEPARABILITY ANALYSIS:")
            summary.append(f"  Can predict task from features: {task_sep['task_prediction_accuracy']:.3f} ± {task_sep['task_prediction_std']:.3f}")
            summary.append(f"  Random baseline: {random_baseline:.3f}")
            
            if task_sep['task_prediction_accuracy'] > random_baseline + 0.2:
                summary.append("  → High task separability (potential catastrophic forgetting)")
            elif task_sep['task_prediction_accuracy'] > random_baseline + 0.1:
                summary.append("  → Moderate task separability")
            else:
                summary.append("  → Low task separability (good feature sharing)")
            summary.append("")
        
        # Inter-task class analysis
        if 'inter_task_analysis' in combined_sep and 'class_task_separability' in combined_sep['inter_task_analysis']:
            inter_task = combined_sep['inter_task_analysis']
            if inter_task['classes_in_multiple_tasks'] > 0:
                summary.append("INTER-TASK CLASS ANALYSIS:")
                summary.append(f"  Classes appearing in multiple tasks: {inter_task['classes_in_multiple_tasks']}")
                
                class_task_sep = inter_task['class_task_separability']
                class_accs = [result['accuracy'] for result in class_task_sep.values() if 'accuracy' in result]
                if class_accs:
                    mean_class_task_acc = np.mean(class_accs)
                    summary.append(f"  Average per-class task prediction accuracy: {mean_class_task_acc:.3f}")
                    
                    if mean_class_task_acc > 0.8:
                        summary.append("  → Strong task-specific class representations")
                    elif mean_class_task_acc > 0.6:
                        summary.append("  → Moderate task-specific class representations")
                    else:
                        summary.append("  → Weak task-specific class representations (good generalization)")
                summary.append("")
    
    # Combined Gaussian analysis
    if combined_gaussian and -1 in combined_gaussian:
        summary.append("COMBINED GAUSSIAN ANALYSIS:")
        combined_gauss = combined_gaussian[-1]
        
        combined_shapiro_fractions = []
        combined_ks_fractions = []
        
        for class_id, class_result in combined_gauss.items():
            if 'feature_tests' in class_result:
                combined_shapiro_fractions.append(class_result['feature_tests']['shapiro_fraction_normal'])
                combined_ks_fractions.append(class_result['feature_tests']['ks_fraction_normal'])
        
        if combined_shapiro_fractions:
            mean_combined_shapiro = np.mean(combined_shapiro_fractions)
            mean_combined_ks = np.mean(combined_ks_fractions)
            
            summary.append(f"  Combined Shapiro-Wilk pass rate: {mean_combined_shapiro:.1%}")
            summary.append(f"  Combined KS test pass rate: {mean_combined_ks:.1%}")
            
            if mean_combined_shapiro > 0.7:
                combined_gaussian_level = "Strong combined Gaussian assumption"
            elif mean_combined_shapiro > 0.5:
                combined_gaussian_level = "Moderate combined Gaussian assumption"
            elif mean_combined_shapiro > 0.3:
                combined_gaussian_level = "Weak combined Gaussian assumption"
            else:
                combined_gaussian_level = "Poor combined Gaussian assumption"
            
            summary.append(f"  → {combined_gaussian_level}")
            summary.append("")

    # Summary Recommendations
    summary.append("RECOMMENDATIONS:")
    summary.append("-" * 40)
    
    # Individual task recommendations
    if separability_results:
        all_svm_accs = [separability_results[tid]['svm_accuracy'] for tid in task_ids if tid in separability_results]
        if all_svm_accs:
            mean_svm_acc = np.mean(all_svm_accs)
            if mean_svm_acc > 0.8:
                summary.append("✓ Individual task features show good linear separability")
            elif mean_svm_acc > 0.6:
                summary.append("△ Individual task features show moderate linear separability")
            else:
                summary.append("✗ Individual task features show poor linear separability")
    
    # Combined feature recommendations
    if combined_separability and -1 in combined_separability and isinstance(combined_separability[-1], dict):
        combined_acc = combined_separability[-1]['svm_accuracy']
        if combined_acc > 0.8:
            summary.append("✓ Combined features show good linear separability - global linear classifiers recommended")
        elif combined_acc > 0.6:
            summary.append("△ Combined features show moderate linear separability - consider ensemble methods")
        else:
            summary.append("✗ Combined features show poor linear separability - non-linear global methods recommended")
        
        # Task separability implications
        if 'task_separability' in combined_separability[-1]:
            task_pred_acc = combined_separability[-1]['task_separability'].get('task_prediction_accuracy', 0)
            n_tasks = combined_separability[-1]['task_separability'].get('n_tasks', 1)
            random_baseline = 1.0 / n_tasks
            
            if task_pred_acc > random_baseline + 0.2:
                summary.append("⚠ High task separability detected - potential catastrophic forgetting")
                summary.append("  → Consider regularization techniques or task-agnostic training")
            elif task_pred_acc > random_baseline + 0.1:
                summary.append("△ Moderate task separability - monitor for catastrophic forgetting")
            else:
                summary.append("✓ Low task separability - good feature sharing across tasks")
    
    # Gaussian recommendations
    if gaussian_results and all_shapiro_fractions:
        mean_gaussian = np.mean(all_shapiro_fractions)
        if mean_gaussian > 0.5:
            summary.append("✓ Features reasonably follow Gaussian distributions - parametric methods applicable")
        else:
            summary.append("△ Features deviate significantly from Gaussian - consider non-parametric methods")
    
    if combined_gaussian and -1 in combined_gaussian and isinstance(combined_gaussian[-1], dict):
        combined_gauss_fracs = [combined_gaussian[-1][cid]['feature_tests']['shapiro_fraction_normal'] 
                               for cid in combined_gaussian[-1] if 'feature_tests' in combined_gaussian[-1][cid]]
        if combined_gauss_fracs:
            mean_combined_gauss = np.mean(combined_gauss_fracs)
            if mean_combined_gauss > 0.5:
                summary.append("✓ Combined features follow Gaussian distributions - global parametric methods suitable")
            else:
                summary.append("△ Combined features deviate from Gaussian - non-parametric global methods recommended")
    
    summary.append("")
    summary.append("=" * 80)
    
    # Save to file
    with open(os.path.join(save_dir, 'comprehensive_analysis_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    # Print to console
    print('\n'.join(summary))

def main():
    parser = argparse.ArgumentParser(description='PCA Analysis for RanPAC Features_h')
    parser.add_argument('--exp_path', type=str, required=True,
                       help='Path to experiment directory (e.g., logs/model_name/dataset/init_cls/increment/exp_name)')
    parser.add_argument('--n_components', type=int, default=50,
                       help='Number of PCA components to compute (default: 50)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots and analysis (default: exp_path/pca_analysis)')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of PCA even if cache exists')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable PCA result caching')
    parser.add_argument('--cache_file', type=str, default=None,
                       help='Custom cache file path (default: output_dir/pca_cache.pkl)')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from existing cache (no PCA computation)')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.exp_path, 'pca_analysis')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set up cache path
    cache_path = None
    if not args.no_cache:
        if args.cache_file is not None:
            cache_path = args.cache_file
        else:
            cache_path = os.path.join(output_dir, 'pca_cache.pkl')
    
    try:
        # Handle plot-only mode
        if args.plot_only:
            if cache_path and os.path.exists(cache_path):
                print("Plot-only mode: Loading existing PCA results...")
                pca_results, task_data, metadata = load_pca_results(cache_path)
                if pca_results is None:
                    raise ValueError("Failed to load cached PCA results")
            else:
                raise ValueError("Plot-only mode requires existing cache file")
        else:
            # Load features from experiment
            print("Loading features from experiment...")
            task_data = load_features_from_experiment(args.exp_path)
            
            if not task_data:
                raise ValueError("No task data loaded!")
            
            # Perform PCA analysis (with caching)
            print("Performing PCA analysis...")
            pca_results = perform_pca_analysis(
                task_data, 
                args.n_components, 
                cache_path=cache_path,
                force_recompute=args.force_recompute
            )
        
        # Create visualizations
        print("Creating visualizations...")
        plot_explained_variance(pca_results, output_dir)
        plot_2d_embeddings(task_data, pca_results, output_dir)
        
        # Analyze cross-task similarity
        print("Analyzing cross-task similarity...")
        similarity_matrix = analyze_cross_task_similarity(pca_results, output_dir)
        
        # Perform linear separability analysis
        print("Analyzing linear separability...")
        separability_results = analyze_linear_separability(task_data, output_dir)
        
        # Perform Gaussian distribution analysis
        print("Analyzing Gaussian assumptions...")
        gaussian_results = analyze_gaussian_assumption(task_data, output_dir)
        
        # Perform combined features analysis
        print("Analyzing combined features from all tasks...")
        combined_separability, combined_gaussian = analyze_combined_features(task_data, output_dir)
        
        # Create additional analysis plots
        print("Creating separability analysis plots...")
        plot_separability_analysis(separability_results, output_dir)
        
        print("Creating Gaussian analysis plots...")
        plot_gaussian_analysis(gaussian_results, output_dir)
        
        print("Creating combined features analysis plots...")
        plot_combined_analysis(combined_separability, combined_gaussian, task_data, output_dir)
        
        # Save comprehensive summary
        print("Saving comprehensive analysis summary...")
        save_analysis_summary(task_data, pca_results, similarity_matrix, 
                             separability_results, gaussian_results, output_dir,
                             combined_separability, combined_gaussian)
        
        if cache_path and not args.no_cache:
            print(f"PCA cache saved to: {cache_path}")
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 