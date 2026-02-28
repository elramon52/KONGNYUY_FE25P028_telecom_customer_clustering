"""
Model selection module for Telecom Customer Segmentation.
Evaluates K-Means, Agglomerative Clustering, and DBSCAN on validation set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .config import (
    KMEANS_K_RANGE, AGGLOMERATIVE_K_RANGE, AGGLOMERATIVE_LINKAGES,
    DBSCAN_EPS_VALUES, DBSCAN_MIN_SAMPLES, COLOR_PALETTE,
    MIN_BUSINESS_CLUSTERS, MAX_BUSINESS_CLUSTERS, FIGURES_DIR
)
from .utils import save_figure, print_header, print_section, setup_plot_style


def evaluate_kmeans(X: np.ndarray, k_range: range = KMEANS_K_RANGE, 
                    random_state: int = 42) -> Dict[str, List]:
    """
    Evaluate K-Means for different values of K with detailed terminal output.
    
    K-Means Algorithm Explanation:
    ------------------------------
    K-Means partitions n observations into k clusters by minimizing within-cluster variance.
    
    Mathematical Objective:
        J = Σ(i=1 to n) Σ(k=1 to K) r_ik ||x_i - μ_k||²
    
    Where:
        - r_ik = 1 if point i belongs to cluster k, else 0
        - x_i = feature vector of observation i
        - μ_k = centroid (mean) of cluster k
        - ||x_i - μ_k||² = squared Euclidean distance
    
    Algorithm Steps:
        1. Initialize k centroids (k-means++ for smart initialization)
        2. Assign each point to nearest centroid
        3. Update centroids as mean of assigned points
        4. Repeat until convergence
    
    Time Complexity: O(n × k × d × i)
        n = number of samples, k = clusters, d = features, i = iterations
    
    Args:
        X: Feature matrix (validation set)
        k_range: Range of K values to test
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with metrics for each K
    """
    print_section("K-MEANS CLUSTERING EVALUATION")
    print("\nAlgorithm: K-Means with k-means++ initialization")
    print("Objective: Minimize within-cluster sum of squares")
    print("Formula: J = Σ(i=1 to n) Σ(k=1 to K) r_ik ||x_i - μ_k||²")
    print(f"\nEvaluating K values from {min(k_range)} to {max(k_range)}...")
    print("=" * 80)
    
    results = {
        'k': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []
    }
    
    print(f"\n{'K':<5} {'Silhouette':<15} {'Davies-Bouldin':<18} {'Calinski-Harabasz':<20} {'Inertia':<15}")
    print("-" * 80)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_
        
        results['k'].append(k)
        results['silhouette'].append(sil)
        results['davies_bouldin'].append(db)
        results['calinski_harabasz'].append(ch)
        results['inertia'].append(inertia)
        
        print(f"{k:<5} {sil:<15.4f} {db:<18.4f} {ch:<20.2f} {inertia:<15.2f}")
    
    print("=" * 80)
    
    # Find optimal K by silhouette
    best_k_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_k_idx]
    best_sil = results['silhouette'][best_k_idx]
    
    print(f"\nK-Means Results Summary:")
    print(f"  Optimal K by Silhouette Score: K={best_k} (Score={best_sil:.4f})")
    print(f"  Interpretation: Higher silhouette = better defined clusters")
    print(f"  Range: [-1, 1] where 1 = perfect clustering, 0 = overlapping, -1 = wrong assignment")
    
    return results


def evaluate_agglomerative(X: np.ndarray, k_range: range = AGGLOMERATIVE_K_RANGE,
                           linkages: List[str] = AGGLOMERATIVE_LINKAGES) -> Dict[str, Any]:
    """
    Evaluate Agglomerative Clustering for different K and linkages.
    
    Agglomerative Clustering Explanation:
    -------------------------------------
    Bottom-up hierarchical clustering that builds a dendrogram.
    
    Algorithm Steps:
        1. Start with n clusters (each point is its own cluster)
        2. Compute pairwise distances between all clusters
        3. Merge closest pair of clusters
        4. Update distance matrix
        5. Repeat until k clusters remain
    
    Linkage Methods:
        - Ward: Minimizes variance increase when merging
        - Complete: Maximum distance between clusters
        - Average: Average distance between clusters
    
    Time Complexity: O(n³) or O(n² log n) with optimizations
    Space Complexity: O(n²) for distance matrix
    
    Args:
        X: Feature matrix
        k_range: Range of K values to test
        linkages: List of linkage methods to test
        
    Returns:
        Dictionary with results for each linkage
    """
    print_section("AGGLOMERATIVE HIERARCHICAL CLUSTERING EVALUATION")
    print("\nAlgorithm: Bottom-up hierarchical clustering")
    print("Approach: Iteratively merge closest clusters until K clusters remain")
    print("\nLinkage Methods Explained:")
    print("  - Ward: Minimizes variance increase (best for K-Means-like objectives)")
    print("  - Complete: Uses maximum distance (produces compact clusters)")
    print("  - Average: Uses average distance (balanced approach)")
    print(f"\nEvaluating K values from {min(k_range)} to {max(k_range)} for each linkage...")
    
    all_results = {}
    
    for linkage in linkages:
        print(f"\n{'='*60}")
        print(f"LINKAGE METHOD: {linkage.upper()}")
        print(f"{'='*60}")
        
        results = {
            'k': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        print(f"\n{'K':<5} {'Silhouette':<15} {'Davies-Bouldin':<18} {'Calinski-Harabasz':<20}")
        print("-" * 60)
        
        for k in k_range:
            if linkage == 'ward':
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric='euclidean')
            else:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric='euclidean')
            
            labels = agg.fit_predict(X)
            
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            results['k'].append(k)
            results['silhouette'].append(sil)
            results['davies_bouldin'].append(db)
            results['calinski_harabasz'].append(ch)
            
            print(f"{k:<5} {sil:<15.4f} {db:<18.4f} {ch:<20.2f}")
        
        all_results[linkage] = results
        
        best_idx = np.argmax(results['silhouette'])
        print(f"\n  Best K for {linkage} linkage: K={results['k'][best_idx]} "
              f"(Silhouette={results['silhouette'][best_idx]:.4f})")
    
    # Find overall best linkage
    best_linkage = None
    best_sil = -1
    for linkage, res in all_results.items():
        max_sil = max(res['silhouette'])
        if max_sil > best_sil:
            best_sil = max_sil
            best_linkage = linkage
    
    print(f"\n{'='*60}")
    print(f"AGGLOMERATIVE SUMMARY:")
    print(f"  Best Linkage Method: {best_linkage}")
    print(f"  Best Silhouette Score: {best_sil:.4f}")
    print(f"{'='*60}")
    
    return all_results


def evaluate_dbscan(X: np.ndarray, eps_values: List[float] = DBSCAN_EPS_VALUES,
                    min_samples_values: List[int] = DBSCAN_MIN_SAMPLES) -> Dict[str, Any]:
    """
    Evaluate DBSCAN for different eps and min_samples values.
    
    DBSCAN (Density-Based Spatial Clustering) Explanation:
    ------------------------------------------------------
    Discovers clusters based on density, can identify arbitrarily shaped clusters.
    
    Key Concepts:
        - Core Point: Has at least min_samples within eps radius
        - Border Point: Within eps of core point but not core itself
        - Noise Point: Neither core nor border (outlier)
    
    Mathematical Definition:
        N_eps(p) = {q ∈ D : dist(p,q) ≤ eps}
        p is core point if |N_eps(p)| ≥ min_samples
    
    Algorithm Steps:
        1. Find all core points
        2. Connect core points within eps of each other
        3. Assign border points to nearby clusters
        4. Mark noise points as -1
    
    Time Complexity: O(n log n) with spatial indexing, O(n²) worst case
    Space Complexity: O(n)
    
    Args:
        X: Feature matrix
        eps_values: List of epsilon values (neighborhood radius)
        min_samples_values: List of min_samples (density threshold)
        
    Returns:
        Dictionary with results grid
    """
    print_section("DBSCAN (DENSITY-BASED CLUSTERING) EVALUATION")
    print("\nAlgorithm: Density-Based Spatial Clustering of Applications with Noise")
    print("Key Parameters:")
    print("  - eps (ε): Neighborhood radius")
    print("  - min_samples: Minimum points to form a dense region")
    print("\nKey Concepts:")
    print("  - Core Point: Has ≥ min_samples neighbors within eps")
    print("  - Border Point: Within eps of core point")
    print("  - Noise Point: Neither core nor border (outlier)")
    print(f"\nGrid Search: eps ∈ {eps_values}, min_samples ∈ {min_samples_values}")
    print("=" * 80)
    
    results = {
        'eps': [],
        'min_samples': [],
        'n_clusters': [],
        'silhouette': [],
        'n_noise': [],
        'noise_pct': []
    }
    
    print(f"\n{'eps':<8} {'min_samp':<10} {'clusters':<10} {'silhouette':<15} {'noise_pts':<12} {'noise_%':<10}")
    print("-" * 80)
    
    best_silhouette = -1
    best_params = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_pct = n_noise / len(labels) * 100
            
            # Compute silhouette only if valid clusters found (ignoring noise)
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    sil_score = silhouette_score(X[mask], labels[mask])
                else:
                    sil_score = -1
            else:
                sil_score = -1
            
            results['eps'].append(eps)
            results['min_samples'].append(min_samples)
            results['n_clusters'].append(n_clusters)
            results['silhouette'].append(sil_score)
            results['n_noise'].append(n_noise)
            results['noise_pct'].append(noise_pct)
            
            sil_str = f"{sil_score:.4f}" if sil_score > 0 else "N/A"
            print(f"{eps:<8.1f} {min_samples:<10} {n_clusters:<10} {sil_str:<15} {n_noise:<12} {noise_pct:<10.1f}")
            
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_params = {'eps': eps, 'min_samples': min_samples, 
                              'n_clusters': n_clusters}
    
    print("=" * 80)
    results['best_silhouette'] = best_silhouette
    results['best_params'] = best_params
    
    if best_params:
        print(f"\nDBSCAN Results Summary:")
        print(f"  Best Parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
        print(f"  Clusters Found: {best_params['n_clusters']}")
        print(f"  Best Silhouette Score: {best_silhouette:.4f}")
    else:
        print("\nDBSCAN Results Summary:")
        print("  No valid clusters found with tested parameters")
    
    return results


def plot_silhouette_vs_k(kmeans_results: Dict, agglomerative_results: Dict,
                         figures_dir: Path = FIGURES_DIR) -> Tuple[str, str]:
    """
    Plot silhouette score vs K for K-Means and Agglomerative.
    
    Returns:
        Tuple of (kmeans_plot_path, agglomerative_plot_path)
    """
    setup_plot_style()
    
    # K-Means silhouette plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(kmeans_results['k'], kmeans_results['silhouette'], 
            marker='o', linewidth=2.5, markersize=10, color='#E63946', label='K-Means')
    
    # Highlight best K
    best_k_idx = np.argmax(kmeans_results['silhouette'])
    best_k = kmeans_results['k'][best_k_idx]
    best_sil = kmeans_results['silhouette'][best_k_idx]
    ax.plot(best_k, best_sil, marker='*', markersize=20, color='#FFD700', 
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.annotate(f'Optimal K={best_k}\nSilhouette={best_sil:.3f}', 
                xy=(best_k, best_sil), xytext=(best_k+0.7, best_sil+0.015),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
    ax.set_title('Silhouette Score vs K for K-Means (Validation Set)\nHigher is Better', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(list(kmeans_results['k']))
    ax.set_ylim(0, max(kmeans_results['silhouette']) * 1.15)
    
    # Add interpretation note
    ax.text(0.02, 0.02, 
            'Silhouette Score Interpretation:\n'
            '  > 0.5: Strong structure\n'
            '  0.25-0.5: Reasonable structure\n'
            '  < 0.25: Weak/no structure',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    kmeans_path = save_figure(fig, '13_silhouette_vs_k_kmeans', figures_dir)
    
    # Agglomerative silhouette plot (best linkage)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    best_linkage = None
    best_avg_sil = -1
    
    colors = {'ward': '#0066CC', 'complete': '#009900', 'average': '#CC6600'}
    
    for linkage, results in agglomerative_results.items():
        avg_sil = np.mean(results['silhouette'])
        if avg_sil > best_avg_sil:
            best_avg_sil = avg_sil
            best_linkage = linkage
        ax.plot(results['k'], results['silhouette'], 
                marker='s', linewidth=2, markersize=8, 
                color=colors.get(linkage, '#333333'), label=f'{linkage.capitalize()} Linkage')
    
    results = agglomerative_results[best_linkage]
    best_k_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_k_idx]
    best_sil = results['silhouette'][best_k_idx]
    ax.plot(best_k, best_sil, marker='*', markersize=20, color='#FFD700',
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.annotate(f'Best K={best_k}\nSil={best_sil:.3f}', 
                xy=(best_k, best_sil), xytext=(best_k+0.7, best_sil+0.015),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Silhouette Score vs K for Agglomerative Clustering (Validation Set)\n'
                 f'Best Linkage: {best_linkage.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(list(results['k']))
    ax.set_ylim(0, max(results['silhouette']) * 1.15)
    
    agg_path = save_figure(fig, '14_silhouette_vs_k_agglomerative', figures_dir)
    
    return kmeans_path, agg_path


def plot_dbscan_heatmap(dbscan_results: Dict, figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot DBSCAN silhouette score heatmap.
    
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    # Create pivot table for heatmap
    df_results = pd.DataFrame({
        'eps': dbscan_results['eps'],
        'min_samples': dbscan_results['min_samples'],
        'silhouette': dbscan_results['silhouette']
    })
    
    pivot = df_results.pivot(index='min_samples', columns='eps', values='silhouette')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Silhouette Score', 'shrink': 0.8},
                linewidths=1, linecolor='gray', annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_xlabel('Epsilon (eps) - Neighborhood Radius', fontsize=13, fontweight='bold')
    ax.set_ylabel('Min Samples - Density Threshold', fontsize=13, fontweight='bold')
    ax.set_title('DBSCAN Silhouette Score Heatmap (Validation Set)\n'
                 'Higher Values (Green) Indicate Better Clustering', 
                 fontsize=14, fontweight='bold')
    
    # Highlight best
    if dbscan_results['best_params']:
        best_eps = dbscan_results['best_params']['eps']
        best_min = dbscan_results['best_params']['min_samples']
        best_sil = dbscan_results['best_silhouette']
        
        # Find position in heatmap
        eps_cols = list(pivot.columns)
        min_rows = list(pivot.index)
        col_idx = eps_cols.index(best_eps)
        row_idx = min_rows.index(best_min)
        
        # Add rectangle around best cell
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, 
                                    fill=False, edgecolor='blue', 
                                    linewidth=4, linestyle='--'))
        
        ax.text(0.5, -0.12, 
                f'★ BEST: eps={best_eps}, min_samples={best_min}, Silhouette={best_sil:.3f} ★',
                transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.9))
    
    return save_figure(fig, '15_dbscan_silhouette_heatmap', figures_dir)


def plot_elbow_method(kmeans_results: Dict, figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot elbow curve for K-Means.
    
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(kmeans_results['k'], kmeans_results['inertia'], 
            marker='o', linewidth=2.5, markersize=10, color='#E63946')
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=13, fontweight='bold')
    ax.set_title('Elbow Method for K-Means (Validation Set)\n'
                 'Look for "Elbow" Point Where Curve Bends', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(list(kmeans_results['k']))
    
    # Add annotation explaining elbow method
    ax.text(0.98, 0.98, 
            'Elbow Method Interpretation:\n'
            'The "elbow" is the point where adding\n'
            'more clusters yields diminishing returns.\n'
            'It represents a trade-off between\n'
            'model complexity and explained variance.',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return save_figure(fig, '16_elbow_method_kmeans', figures_dir)


def plot_algorithm_comparison(kmeans_results: Dict, agglomerative_results: Dict,
                              dbscan_results: Dict, figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot comparison of best silhouette scores across algorithms.
    
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    # Get best scores
    kmeans_best = max(kmeans_results['silhouette'])
    kmeans_best_k = kmeans_results['k'][np.argmax(kmeans_results['silhouette'])]
    
    best_linkage = None
    best_agg_score = -1
    for linkage, results in agglomerative_results.items():
        best_for_linkage = max(results['silhouette'])
        if best_for_linkage > best_agg_score:
            best_agg_score = best_for_linkage
            best_linkage = linkage
    agg_best_k = agglomerative_results[best_linkage]['k'][np.argmax(agglomerative_results[best_linkage]['silhouette'])]
    
    dbscan_best = dbscan_results['best_silhouette'] if dbscan_results['best_silhouette'] > 0 else 0
    dbscan_k = dbscan_results['best_params']['n_clusters'] if dbscan_results['best_params'] else 0
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algorithms = [f'K-Means\n(K={kmeans_best_k})', 
                  f'Agglomerative\n({best_linkage}, K={agg_best_k})', 
                  f'DBSCAN\n(K={dbscan_k})']
    scores = [kmeans_best, best_agg_score, dbscan_best]
    colors = ['#E63946', '#0066CC', '#009900']
    
    bars = ax.bar(algorithms, scores, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        label = f'{score:.3f}' if score > 0 else 'N/A'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Highlight best algorithm
    best_idx = np.argmax(scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    ax.set_ylabel('Best Silhouette Score', fontsize=13, fontweight='bold')
    ax.set_title('Algorithm Comparison: Best Silhouette Score (Validation Set)\n'
                 'Gold Border Indicates Best Overall Algorithm', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add interpretation note
    if dbscan_best <= 0:
        ax.text(0.5, 0.15, 
                'Note: DBSCAN did not find valid clusters\n'
                'with the tested parameter combinations.\n'
                'This suggests the data may not have\n'
                'well-separated density regions.',
                transform=ax.transAxes, fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    return save_figure(fig, '17_algorithm_comparison', figures_dir)


def apply_business_override(kmeans_results: Dict, 
                            agglomerative_results: Dict) -> Tuple[str, int, Dict]:
    """
    Apply business-driven selection criteria to choose optimal K.
    
    Business-Driven Selection Rationale:
    ------------------------------------
    While mathematical metrics guide initial selection, business considerations
    may override purely mathematical optima when they conflict with operational needs.
    
    Business Rules:
        1. K=1 is invalid - no segmentation possible
        2. K=2 is too trivial - only separates high/low value (any analyst could do this)
        3. K>6 is too granular - operationally difficult to manage >6 distinct segments
    
    Domain Knowledge Integration:
        - Telecom customers typically fall into lifecycle stages (new, established, loyal)
        - Contract types create natural groupings (month-to-month vs. committed)
        - Service usage patterns differentiate engagement levels
        - K=3-5 typically captures these natural divisions
    
    Args:
        kmeans_results: K-Means evaluation results
        agglomerative_results: Agglomerative evaluation results
        
    Returns:
        Tuple of (algorithm_name, optimal_k, selection_details)
    """
    print_section("BUSINESS-DRIVEN MODEL SELECTION")
    print("\n" + "="*80)
    print("APPLYING BUSINESS RULES TO SELECT FINAL MODEL")
    print("="*80)
    
    # Find mathematically optimal K from K-Means
    best_kmeans_k = kmeans_results['k'][np.argmax(kmeans_results['silhouette'])]
    best_kmeans_sil = max(kmeans_results['silhouette'])
    
    print(f"\nMathematical Analysis:")
    print(f"  K-Means optimal K: {best_kmeans_k} (Silhouette: {best_kmeans_sil:.4f})")
    
    # Find best from Agglomerative
    best_linkage = None
    best_agg_k = None
    best_agg_sil = -1
    for linkage, res in agglomerative_results.items():
        best_idx = np.argmax(res['silhouette'])
        if res['silhouette'][best_idx] > best_agg_sil:
            best_agg_sil = res['silhouette'][best_idx]
            best_agg_k = res['k'][best_idx]
            best_linkage = linkage
    
    print(f"  Agglomerative optimal: K={best_agg_k} with {best_linkage} linkage (Silhouette: {best_agg_sil:.4f})")
    
    # Apply business rules
    selection_details = {
        'mathematical_optimum_k': best_kmeans_k,
        'mathematical_optimum_sil': best_kmeans_sil,
        'business_override': False,
        'override_reason': None,
        'final_k': best_kmeans_k,
        'final_algorithm': 'K-Means'
    }
    
    print(f"\nBusiness Rules Application:")
    print(f"  Rule 1: K=1 is INVALID (no segmentation possible)")
    print(f"  Rule 2: K=2 is TOO TRIVIAL (only high/low value split)")
    print(f"  Rule 3: K>6 is TOO GRANULAR (operationally difficult)")
    
    # Rule 1: K=1 is invalid
    if best_kmeans_k == 1:
        for k, sil in zip(kmeans_results['k'], kmeans_results['silhouette']):
            if k >= MIN_BUSINESS_CLUSTERS:
                selection_details['final_k'] = k
                selection_details['business_override'] = True
                selection_details['override_reason'] = (
                    f"K=1 provides no segmentation. Selected K={k} for actionable business segments."
                )
                break
    
    # Rule 2: K=2 is too trivial for telecom
    elif best_kmeans_k == 2:
        print(f"\n  ⚠ OVERRIDE APPLIED: K=2 is mathematically optimal but TOO TRIVIAL")
        print(f"     Reason: Only separates high/low value customers - any business analyst could do this")
        
        # Find next best K >= 3 with similar score
        for k, sil in zip(kmeans_results['k'], kmeans_results['silhouette']):
            if k >= MIN_BUSINESS_CLUSTERS:
                score_diff = best_kmeans_sil - sil
                if score_diff <= 0.05:  # Within 0.05 of best
                    selection_details['final_k'] = k
                    selection_details['business_override'] = True
                    selection_details['override_reason'] = (
                        f"While K=2 yields highest silhouette ({best_kmeans_sil:.3f}), it only separates "
                        f"high/low value customers. K={k} provides more actionable segments with "
                        f"minimal performance drop (Δ={score_diff:.3f}). "
                        f"Domain knowledge: Telecom customers typically fall into lifecycle stages "
                        f"(new, established, loyal) which K={k} better captures."
                    )
                    break
        
        # If no K within threshold, select K=3
        if not selection_details['business_override']:
            selection_details['final_k'] = 3
            selection_details['business_override'] = True
            selection_details['override_reason'] = (
                f"K=2 is too trivial for telecom segmentation. Selected K=3 to enable "
                f"targeted marketing for distinct customer lifecycle stages (new, established, loyal). "
                f"Domain knowledge: Three segments align with typical telecom customer journeys."
            )
    
    # Rule 3: K > 6 is too granular
    elif best_kmeans_k > MAX_BUSINESS_CLUSTERS:
        print(f"\n  ⚠ OVERRIDE APPLIED: K={best_kmeans_k} is TOO GRANULAR")
        print(f"     Reason: Managing >6 segments operationally difficult")
        
        best_valid_k = None
        best_valid_sil = -1
        for k, sil in zip(kmeans_results['k'], kmeans_results['silhouette']):
            if k <= MAX_BUSINESS_CLUSTERS and sil > best_valid_sil:
                best_valid_k = k
                best_valid_sil = sil
        
        selection_details['final_k'] = best_valid_k
        selection_details['business_override'] = True
        selection_details['override_reason'] = (
            f"K={best_kmeans_k} is too granular for operational marketing. "
            f"Selected K={best_valid_k} for practical business utility. "
            f"Domain knowledge: Marketing teams can effectively manage 3-6 segments."
        )
    
    print(f"\n{'='*80}")
    print("FINAL SELECTION:")
    print(f"{'='*80}")
    print(f"  Algorithm: {selection_details['final_algorithm']}")
    print(f"  Number of Clusters (K): {selection_details['final_k']}")
    
    if selection_details['business_override']:
        print(f"\n  BUSINESS OVERRIDE APPLIED:")
        print(f"  {selection_details['override_reason']}")
    else:
        print(f"\n  No override needed - mathematical optimum aligns with business needs")
    
    print(f"{'='*80}")
    
    return (selection_details['final_algorithm'], 
            selection_details['final_k'], 
            selection_details)


def run_model_selection(X_train: np.ndarray, X_val: np.ndarray,
                        figures_dir: Path = FIGURES_DIR) -> Dict[str, Any]:
    """
    Run complete model selection process with detailed terminal output.
    
    Args:
        X_train: Training features (for context)
        X_val: Validation features (for evaluation)
        figures_dir: Directory to save figures
        
    Returns:
        Dictionary with all results and selected model info
    """
    print_header("MODEL SELECTION (Validation Set)")
    print("\n" + "="*80)
    print("EVALUATING THREE CLUSTERING ALGORITHMS")
    print("="*80)
    print("\nAlgorithms to Evaluate:")
    print("  1. K-Means: Partition-based clustering")
    print("  2. Agglomerative Hierarchical Clustering")
    print("  3. DBSCAN: Density-based clustering")
    print("\nValidation Metrics:")
    print("  - Silhouette Score: Measures cluster cohesion and separation")
    print("  - Davies-Bouldin Index: Ratio of within-cluster scatter to between-cluster separation")
    print("  - Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion")
    
    results = {}
    
    # Evaluate K-Means
    print("\n" + "="*80)
    kmeans_results = evaluate_kmeans(X_val)
    results['kmeans'] = kmeans_results
    
    # Evaluate Agglomerative
    print("\n" + "="*80)
    agg_results = evaluate_agglomerative(X_val)
    results['agglomerative'] = agg_results
    
    # Evaluate DBSCAN
    print("\n" + "="*80)
    dbscan_results = evaluate_dbscan(X_val)
    results['dbscan'] = dbscan_results
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING MODEL SELECTION VISUALIZATIONS")
    print("="*80)
    
    kmeans_plot, agg_plot = plot_silhouette_vs_k(kmeans_results, agg_results, figures_dir)
    print(f"  ✓ Saved: K-Means silhouette plot")
    print(f"  ✓ Saved: Agglomerative silhouette plot")
    
    dbscan_plot = plot_dbscan_heatmap(dbscan_results, figures_dir)
    print(f"  ✓ Saved: DBSCAN parameter heatmap")
    
    elbow_plot = plot_elbow_method(kmeans_results, figures_dir)
    print(f"  ✓ Saved: Elbow method plot")
    
    comparison_plot = plot_algorithm_comparison(kmeans_results, agg_results, 
                                                 dbscan_results, figures_dir)
    print(f"  ✓ Saved: Algorithm comparison plot")
    
    results['plots'] = {
        'kmeans_silhouette': kmeans_plot,
        'agglomerative_silhouette': agg_plot,
        'dbscan_heatmap': dbscan_plot,
        'elbow_method': elbow_plot,
        'algorithm_comparison': comparison_plot
    }
    
    # Apply business override
    print("\n" + "="*80)
    algorithm, optimal_k, selection_details = apply_business_override(
        kmeans_results, agg_results
    )
    
    results['selected'] = {
        'algorithm': algorithm,
        'k': optimal_k,
        'selection_details': selection_details
    }
    
    return results
