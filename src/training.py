"""
Training and evaluation module for Telecom Customer Segmentation.
Handles final model training and test set evaluation with distinct colors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
from typing import Dict, List, Tuple, Any
from pathlib import Path

from .config import FIGURES_DIR, COLOR_PALETTE, RANDOM_STATE, TARGET_COL
from .utils import save_figure, print_header, compute_confidence_interval, cramers_v

# Define distinct colors for clusters - optimized for visibility
# Using deep, saturated colors that are easily distinguishable
DISTINCT_COLORS = {
    0: '#0066CC',  # Deep Blue
    1: '#CC0000',  # Deep Red
    2: '#009900',  # Deep Green
    3: '#FF6600',  # Deep Orange
    4: '#9900CC',  # Deep Purple
    5: '#CC9900',  # Deep Gold
}


def train_final_model(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> KMeans:
    """
    Train final K-Means model.
    
    Args:
        X: Feature matrix
        k: Number of clusters
        random_state: Random seed
        
    Returns:
        Trained KMeans model
    """
    print(f"\nTraining final K-Means model with K={k}...")
    
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(X)
    
    print(f"  ✓ Model trained successfully")
    print(f"  ✓ Inertia (Within-cluster sum of squares): {model.inertia_:.2f}")
    print(f"  ✓ Iterations to convergence: {model.n_iter_}")
    
    return model


def evaluate_model(X: np.ndarray, labels: np.ndarray, 
                   dataset_name: str = "Dataset") -> Dict[str, float]:
    """
    Compute internal validation metrics.
    
    Metrics Explanation:
    --------------------
    Silhouette Score: Measures how similar an object is to its own cluster 
                      compared to other clusters.
        Range: [-1, 1]
        Interpretation: 
            > 0.5: Strong structure
            0.25-0.5: Reasonable structure  
            < 0.25: Weak/no structure
        Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
            a(i) = average distance to points in same cluster
            b(i) = average distance to points in nearest other cluster
    
    Davies-Bouldin Index: Ratio of within-cluster scatter to between-cluster separation.
        Range: [0, ∞)
        Interpretation: Lower is better
        Formula: DB = (1/k) Σ(i=1 to k) max(j≠i) [(σ_i + σ_j) / d(c_i, c_j)]
            σ_i = average distance within cluster i
            d(c_i, c_j) = distance between centroids
    
    Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion.
        Range: [0, ∞)
        Interpretation: Higher is better
        Formula: CH = [tr(B)/(k-1)] / [tr(W)/(n-k)]
            B = between-cluster dispersion matrix
            W = within-cluster dispersion matrix
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        dataset_name: Name of dataset for printing
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'silhouette_score': silhouette_score(X, labels),
        'davies_bouldin_index': davies_bouldin_score(X, labels),
        'calinski_harabasz_index': calinski_harabasz_score(X, labels),
        'n_samples': len(X),
        'n_clusters': len(np.unique(labels))
    }
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Clusters: {metrics['n_clusters']}")
    print(f"\nInternal Validation Metrics:")
    print(f"  1. Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"     Interpretation: ", end="")
    if metrics['silhouette_score'] > 0.5:
        print("Strong cluster structure")
    elif metrics['silhouette_score'] > 0.25:
        print("Reasonable cluster structure")
    else:
        print("Weak cluster structure")
    
    print(f"  2. Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"     Interpretation: Lower is better (compact, well-separated clusters)")
    
    print(f"  3. Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")
    print(f"     Interpretation: Higher is better (distinct clusters)")
    print(f"{'='*60}")
    
    return metrics


def external_validation(df: pd.DataFrame, labels: np.ndarray, 
                        target_col: str = TARGET_COL) -> Dict[str, Any]:
    """
    Perform external validation using churn as ground truth proxy.
    
    External Validation Purpose:
    ----------------------------
    While internal metrics assess clustering quality mathematically, external 
    validation checks if clusters have business meaning. Churn rate differences 
    across clusters indicate that segments represent genuinely different customer types.
    
    Chi-Square Test of Independence:
    --------------------------------
    H0 (Null Hypothesis): Cluster assignment and churn are independent
    H1 (Alternative): Cluster assignment and churn are associated
    
    Test Statistic: χ² = Σ (O - E)² / E
        O = Observed frequency
        E = Expected frequency under independence
    
    Decision Rule:
        If p-value < 0.05: Reject H0 (significant association)
        If p-value ≥ 0.05: Fail to reject H0 (no significant association)
    
    Cramer's V:
    -----------
    Measures effect size (strength of association) between categorical variables.
    Range: [0, 1]
    Interpretation:
        0.1: Small effect
        0.3: Medium effect
        0.5: Large effect
    
    Args:
        df: DataFrame with original data including Churn column
        labels: Cluster labels
        target_col: Name of target column
        
    Returns:
        Dictionary with external validation results
    """
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION (Using Churn as Proxy)")
    print("="*60)
    print("\nPurpose: Validate that clusters have business meaning")
    print("Method: Test association between cluster assignment and churn")
    
    # Convert churn to binary
    if df[target_col].dtype == 'object':
        churn_binary = (df[target_col] == 'Yes').astype(int).values
    else:
        churn_binary = df[target_col].values
    
    overall_churn_rate = churn_binary.mean()
    print(f"\nOverall Churn Rate: {overall_churn_rate*100:.2f}%")
    
    # Churn rate per cluster with confidence intervals
    cluster_churn = {}
    unique_labels = np.unique(labels)
    
    print(f"\n{'Cluster':<10} {'Size':<8} {'Churned':<10} {'Churn Rate':<15} {'95% CI':<25}")
    print("-" * 70)
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = churn_binary[mask]
        
        n_samples = len(cluster_data)
        n_churn = cluster_data.sum()
        churn_rate = n_churn / n_samples if n_samples > 0 else 0
        
        # Compute 95% confidence interval
        ci_lower, ci_upper = compute_confidence_interval(n_churn, n_samples)
        
        cluster_churn[cluster_id] = {
            'n_samples': n_samples,
            'n_churn': n_churn,
            'churn_rate': churn_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        print(f"{cluster_id:<10} {n_samples:<8} {n_churn:<10} "
              f"{churn_rate*100:<14.2f} [{ci_lower*100:.1f}% - {ci_upper*100:.1f}%]")
    
    # Chi-square test of independence
    print("\n" + "-" * 60)
    print("Chi-Square Test of Independence")
    print("-" * 60)
    print("H0: Cluster assignment and churn are independent")
    print("H1: Cluster assignment and churn are associated")
    
    contingency = pd.crosstab(labels, churn_binary)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Cramer's V
    cramers_v_value = cramers_v(contingency.values)
    
    print(f"\nTest Results:")
    print(f"  χ² statistic: {chi2:.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Cramer's V: {cramers_v_value:.3f}")
    
    print(f"\nInterpretation:")
    if p_value < 0.001:
        print(f"  ✓ HIGHLY SIGNIFICANT association (p < 0.001)")
        print(f"  ✓ Clusters have strong business meaning")
    elif p_value < 0.05:
        print(f"  ✓ SIGNIFICANT association (p < 0.05)")
        print(f"  ✓ Clusters have business meaning")
    else:
        print(f"  ✗ No significant association (p ≥ 0.05)")
        print(f"  ✗ Clusters may not have business meaning")
    
    print(f"\nEffect Size (Cramer's V):")
    if cramers_v_value < 0.1:
        print(f"  Negligible effect (V < 0.1)")
    elif cramers_v_value < 0.3:
        print(f"  Small effect (0.1 ≤ V < 0.3)")
    elif cramers_v_value < 0.5:
        print(f"  Medium effect (0.3 ≤ V < 0.5)")
    else:
        print(f"  Large effect (V ≥ 0.5)")
    
    print(f"{'='*60}")
    
    return {
        'cluster_churn': cluster_churn,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v_value,
        'contingency_table': contingency
    }


def plot_churn_by_cluster(cluster_churn: Dict, cluster_names: Dict[int, str] = None,
                          figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot churn rate by cluster with confidence intervals.
    
    Visualization Purpose:
    ----------------------
    Shows how churn rates differ across clusters, validating that segments
    represent genuinely different customer types with varying risk levels.
    
    Args:
        cluster_churn: Dictionary with churn rates per cluster
        cluster_names: Dictionary mapping cluster_id to name
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    clusters = sorted(cluster_churn.keys())
    churn_rates = [cluster_churn[c]['churn_rate'] * 100 for c in clusters]
    ci_lowers = [cluster_churn[c]['ci_lower'] * 100 for c in clusters]
    ci_uppers = [cluster_churn[c]['ci_upper'] * 100 for c in clusters]
    
    # Error bars
    errors_lower = [cr - ci_l for cr, ci_l in zip(churn_rates, ci_lowers)]
    errors_upper = [ci_u - cr for cr, ci_u in zip(churn_rates, ci_uppers)]
    
    # Labels
    if cluster_names:
        labels = [cluster_names.get(c, f'Cluster {c}') for c in clusters]
    else:
        labels = [f'Cluster {c}' for c in clusters]
    
    # Use distinct colors for each cluster
    colors = [DISTINCT_COLORS.get(i, '#333333') for i in range(len(clusters))]
    
    bars = ax.bar(labels, churn_rates, yerr=[errors_lower, errors_upper],
                  capsize=8, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels on bars
    for bar, rate in zip(bars, churn_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add overall average line
    overall_avg = np.mean(churn_rates)
    ax.axhline(y=overall_avg, color='#FF0000', linestyle='--', linewidth=2.5, 
               label=f'Overall Average ({overall_avg:.1f}%)')
    
    ax.set_ylabel('Churn Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Customer Segment', fontsize=14, fontweight='bold')
    ax.set_title('Churn Rate by Cluster with 95% Confidence Interval\n'
                 'Validates Business Relevance of Segments', 
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(churn_rates) * 1.25)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=12)
    ax.tick_params(axis='x', rotation=15, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Add interpretation note
    ax.text(0.02, 0.98, 
            'Interpretation:\n'
            'Significant differences in churn rates\n'
            'across clusters validate that segments\n'
            'represent genuinely different customer types.',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return save_figure(fig, '18_churn_rate_by_cluster', figures_dir)


def plot_pca_projection(X: np.ndarray, labels: np.ndarray, 
                        cluster_names: Dict[int, str] = None,
                        figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot 2D PCA projection colored by cluster with distinct colors.
    
    PCA (Principal Component Analysis) Explanation:
    -----------------------------------------------
    Dimensionality reduction technique that projects high-dimensional data
    onto a lower-dimensional space while preserving maximum variance.
    
    Mathematical Steps:
        1. Center data: X_centered = X - mean(X)
        2. Compute covariance matrix: C = (1/n) X^T X
        3. Eigen decomposition: C = V Λ V^T
        4. Principal components = eigenvectors corresponding to largest eigenvalues
        5. Project: Z = X V_k (first k components)
    
    Visualization Purpose:
    ----------------------
    Shows how clusters are distributed in the 2D space of maximum variance.
    Well-separated clusters indicate good segmentation quality.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        cluster_names: Dictionary mapping cluster_id to name
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_labels = np.unique(labels)
    
    # Plot each cluster with distinct color
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        color = DISTINCT_COLORS.get(i, '#333333')
        label = cluster_names.get(cluster_id, f'Cluster {cluster_id}') if cluster_names else f'Cluster {cluster_id}'
        
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color, label=label, alpha=0.7, s=50, 
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                  fontsize=13, fontweight='bold')
    ax.set_title('PCA Projection of Test Set by Cluster Assignment\n'
                 '2D Visualization of High-Dimensional Customer Data', 
                 fontsize=15, fontweight='bold')
    ax.legend(title='Customer Segments', title_fontsize=13, fontsize=12, 
              loc='best', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Add variance explained annotation
    total_var = sum(pca.explained_variance_ratio_) * 100
    ax.text(0.02, 0.98, 
            f'Total Variance Explained: {total_var:.1f}%\n\n'
            f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%): '
            f'Primary axis of variation\n'
            f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%): '
            f'Secondary axis of variation',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    return save_figure(fig, '19_pca_projection_test', figures_dir)


def plot_cluster_centroids_heatmap(X: np.ndarray, labels: np.ndarray, 
                                   feature_names: List[str],
                                   figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot heatmap of cluster centroids for top features.
    
    Visualization Purpose:
    ----------------------
    Shows the average value of each feature for each cluster.
    Helps identify which features characterize each segment.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        feature_names: List of feature names
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    # Compute cluster centroids
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Select top 15 features by variance
    feature_vars = np.var(X, axis=0)
    top_feature_indices = np.argsort(feature_vars)[-15:]
    top_feature_names = [feature_names[i] for i in top_feature_indices]
    
    # Compute centroids for top features
    centroids = np.zeros((n_clusters, len(top_feature_indices)))
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        centroids[i] = np.mean(X[mask][:, top_feature_indices], axis=0)
    
    # Z-score normalize centroids for visualization
    centroids_z = (centroids - centroids.mean(axis=0)) / (centroids.std(axis=0) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(centroids_z, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=top_feature_names,
                yticklabels=[f'Cluster {i}' for i in unique_labels],
                ax=ax, cbar_kws={'label': 'Z-score (Standard Deviations from Mean)', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray', annot_kws={'size': 10})
    
    ax.set_title('Cluster Centroids Heatmap (Top 15 Features, Z-scored)\n'
                 'Red = Above Average, Blue = Below Average', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=13, fontweight='bold')
    ax.set_ylabel('Clusters', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=11)
    
    # Add interpretation note
    ax.text(1.15, 0.98, 
            'Interpretation:\n'
            '• Red values: Cluster has HIGHER\n'
            '  than average values for that feature\n'
            '• Blue values: Cluster has LOWER\n'
            '  than average values for that feature\n'
            '• White values: Near average',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return save_figure(fig, '20_cluster_centroids_heatmap', figures_dir)


def plot_silhouette_per_sample(X: np.ndarray, labels: np.ndarray,
                               figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot per-sample silhouette scores sorted within clusters.
    
    Visualization Purpose:
    ----------------------
    Shows the silhouette score for each individual sample, sorted within clusters.
    Wide variation within a cluster suggests heterogeneous membership.
    Negative values indicate potentially misclassified points.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    from sklearn.metrics import silhouette_samples
    
    silhouette_vals = silhouette_samples(X, labels)
    avg_score = silhouette_vals.mean()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    y_lower = 10
    for i, cluster_id in enumerate(unique_labels):
        cluster_silhouette_vals = silhouette_vals[labels == cluster_id]
        cluster_silhouette_vals.sort()
        
        size_cluster = len(cluster_silhouette_vals)
        y_upper = y_lower + size_cluster
        
        color = DISTINCT_COLORS.get(i, '#333333')
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Add cluster label
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id),
                fontsize=12, fontweight='bold', ha='right', va='center')
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=13, fontweight='bold')
    ax.set_title('Per-Sample Silhouette Plot (Sorted Within Clusters)\n'
                 'Higher Values Indicate Better Cluster Membership', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=2.5,
               label=f'Average Silhouette: {avg_score:.3f}')
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(-0.1, 1)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add interpretation note
    ax.text(0.98, 0.98, 
            'Interpretation:\n'
            '• Width of each section = cluster size\n'
            '• Height of filled area = silhouette scores\n'
            '• Red dashed line = average score\n'
            '• Negative values = potential misclassification',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return save_figure(fig, '21_silhouette_per_sample', figures_dir)


def plot_cluster_size_pie(labels: np.ndarray, cluster_names: Dict[int, str] = None,
                          figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot pie chart of cluster sizes.
    
    Visualization Purpose:
    ----------------------
    Shows the proportion of customers in each segment.
    Helps understand the relative importance of each segment.
    
    Args:
        labels: Cluster labels
        cluster_names: Dictionary mapping cluster_id to name
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    if cluster_names:
        labels_pie = [cluster_names.get(l, f'Cluster {l}') for l in unique_labels]
    else:
        labels_pie = [f'Cluster {l}' for l in unique_labels]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Use distinct colors
    colors = [DISTINCT_COLORS.get(i, '#333333') for i in range(len(unique_labels))]
    explode = [0.03] * len(unique_labels)
    
    wedges, texts, autotexts = ax.pie(counts, labels=labels_pie, autopct='%1.1f%%',
                                       colors=colors, explode=explode, shadow=True,
                                       textprops={'fontsize': 12},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    
    # Make percentage labels bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    ax.set_title('Cluster Size Distribution\n'
                 'Proportion of Customers in Each Segment', 
                 fontsize=15, fontweight='bold')
    
    # Add legend with counts
    legend_labels = [f'{l}: {c} customers ({p:.1f}%)' for l, c, p in zip(labels_pie, counts, percentages)]
    ax.legend(wedges, legend_labels, title="Segments", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11, title_fontsize=12)
    
    return save_figure(fig, '22_cluster_size_pie', figures_dir)


def run_final_training_and_evaluation(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    df_test: pd.DataFrame,
    selected_k: int,
    feature_names: List[str],
    figures_dir: Path = FIGURES_DIR
) -> Dict[str, Any]:
    """
    Run complete final training and evaluation pipeline.
    
    Training Protocol:
    ------------------
    Step 1: Train on training set only
        - Fit model on 60% of data
        - Validate on validation set (20%)
    
    Step 2: Retrain on train + validation combined
        - Combine training and validation (80% total)
        - Final model uses maximum available data
    
    Step 3: Evaluate on held-out test set
        - Unbiased evaluation on unseen data (20%)
        - Internal metrics + external validation
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        df_test: Test DataFrame with original columns (including Churn)
        selected_k: Selected number of clusters
        feature_names: List of feature names
        figures_dir: Directory to save figures
        
    Returns:
        Dictionary with all results
    """
    print_header("FINAL MODEL TRAINING AND EVALUATION")
    
    results = {}
    
    # Step 1: Train on training set only
    print("\n" + "="*60)
    print("STEP 1: TRAINING ON TRAINING SET ONLY")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    
    model_train = train_final_model(X_train, selected_k)
    train_labels = model_train.labels_
    
    # Evaluate on validation set
    print("\n" + "-"*60)
    print("VALIDATION SET EVALUATION")
    print("-"*60)
    val_labels = model_train.predict(X_val)
    val_metrics = evaluate_model(X_val, val_labels, "Validation")
    results['validation_metrics'] = val_metrics
    
    print("\n✓ Training on training set completed. Validation metrics printed above.")
    
    # Step 2: Retrain on train + validation combined
    print("\n" + "="*60)
    print("STEP 2: RETRAINING ON TRAIN + VALIDATION COMBINED")
    print("="*60)
    X_train_val = np.vstack([X_train, X_val])
    print(f"Combined training samples: {len(X_train_val)}")
    
    model_final = train_final_model(X_train_val, selected_k)
    
    print("\n✓ Final model trained on train+val. Proceeding to test set evaluation...")
    
    # Step 3: Test set evaluation
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"Test samples: {len(X_test)}")
    print("Note: Test set was held out during model selection and training")
    
    test_labels = model_final.predict(X_test)
    
    # Internal metrics
    test_metrics = evaluate_model(X_test, test_labels, "Test")
    results['test_metrics'] = test_metrics
    
    # External validation
    external_results = external_validation(df_test, test_labels)
    results['external_validation'] = external_results
    
    # Store model and labels
    results['model'] = model_final
    results['test_labels'] = test_labels
    
    return results


def generate_test_evaluation_plots(
    X_test: np.ndarray,
    test_labels: np.ndarray,
    external_results: Dict,
    feature_names: List[str],
    cluster_names: Dict[int, str] = None,
    figures_dir: Path = FIGURES_DIR
) -> Dict[str, str]:
    """
    Generate all test evaluation plots with distinct colors.
    
    Args:
        X_test: Test features
        test_labels: Cluster labels for test set
        external_results: Results from external_validation()
        feature_names: List of feature names
        cluster_names: Dictionary mapping cluster_id to name
        figures_dir: Directory to save figures
        
    Returns:
        Dictionary of plot paths
    """
    print("\n" + "="*60)
    print("GENERATING TEST EVALUATION VISUALIZATIONS")
    print("="*60)
    
    plots = {}
    
    # Plot 18: Churn rate by cluster
    print("\n  Generating: Churn Rate by Cluster...")
    plots['churn_rate'] = plot_churn_by_cluster(
        external_results['cluster_churn'], cluster_names, figures_dir
    )
    print(f"  ✓ Saved: {plots['churn_rate']}")
    
    # Plot 19: PCA projection
    print("\n  Generating: PCA Projection...")
    plots['pca_projection'] = plot_pca_projection(
        X_test, test_labels, cluster_names, figures_dir
    )
    print(f"  ✓ Saved: {plots['pca_projection']}")
    
    # Plot 20: Cluster centroids heatmap
    print("\n  Generating: Cluster Centroids Heatmap...")
    plots['centroids_heatmap'] = plot_cluster_centroids_heatmap(
        X_test, test_labels, feature_names, figures_dir
    )
    print(f"  ✓ Saved: {plots['centroids_heatmap']}")
    
    # Plot 21: Silhouette per sample
    print("\n  Generating: Per-Sample Silhouette Plot...")
    plots['silhouette_per_sample'] = plot_silhouette_per_sample(
        X_test, test_labels, figures_dir
    )
    print(f"  ✓ Saved: {plots['silhouette_per_sample']}")
    
    # Plot 22: Cluster size pie
    print("\n  Generating: Cluster Size Distribution...")
    plots['cluster_size_pie'] = plot_cluster_size_pie(
        test_labels, cluster_names, figures_dir
    )
    print(f"  ✓ Saved: {plots['cluster_size_pie']}")
    
    print("\n✓ All test evaluation visualizations generated successfully")
    
    return plots
