"""
Cluster profiling module for Telecom Customer Segmentation.
Handles cluster characterization, naming, business actions, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
from pathlib import Path

from .config import FIGURES_DIR, COLOR_PALETTE, TARGET_COL
from .utils import save_figure, print_header, print_section


def compute_anova_f_statistics(df: pd.DataFrame, labels: np.ndarray, 
                               feature_names: List[str]) -> pd.DataFrame:
    """
    Compute ANOVA F-statistic for each feature vs cluster assignment.
    
    Args:
        df: DataFrame with features
        labels: Cluster labels
        feature_names: List of feature names to test
        
    Returns:
        DataFrame with F-statistics and p-values
    """
    print_section("ANOVA F-Statistics for Feature Discrimination")
    
    results = []
    
    for feature in feature_names:
        if feature not in df.columns:
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
            
        # Group data by cluster
        groups = [df[feature][labels == cluster].values for cluster in np.unique(labels)]
        
        # Compute ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            
            results.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        except Exception as e:
            print(f"  Warning: Could not compute ANOVA for {feature}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f_statistic', ascending=False)
    
    print("\nTop 10 Discriminating Features:")
    for i, row in results_df.head(10).iterrows():
        sig_marker = "*" if row['significant'] else ""
        print(f"  {row['feature']}: F={row['f_statistic']:.2f}, p={row['p_value']:.2e} {sig_marker}")
    
    return results_df


def characterize_clusters(df: pd.DataFrame, labels: np.ndarray,
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Compute cluster profiles with mean values for key features.
    (Modified to compute service_diversity if not present.)
    """
    print_section("Cluster Characterization")
    
    unique_labels = np.unique(labels)
    profiles = []
    
    # Define service columns for on-the-fly computation
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = df[mask]
        
        profile = {'cluster_id': cluster_id}
        
        # Size
        profile['size'] = len(cluster_data)
        profile['size_pct'] = len(cluster_data) / len(df) * 100
        
        # Churn rate
        if TARGET_COL in df.columns:
            churn_rate = (cluster_data[TARGET_COL] == 'Yes').mean()
            profile['churn_rate'] = churn_rate
        
        # Key numerical features (use original unscaled values if available)
        key_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                       'avg_monthly_spend', 'service_diversity']
        
        for feature in key_features:
            if feature == 'service_diversity' and feature not in df.columns:
                # Compute service_diversity on the fly from raw columns
                # Ensure all service columns exist
                if all(col in df.columns for col in service_cols):
                    # Convert Yes/No to 1/0 and sum
                    service_sum = (cluster_data[service_cols] == 'Yes').sum(axis=1)
                    profile['service_diversity_mean'] = service_sum.mean()
                    profile['service_diversity_std'] = service_sum.std()
                else:
                    # If columns missing, set to 0 (or None)
                    profile['service_diversity_mean'] = 0.0
                    profile['service_diversity_std'] = 0.0
            elif feature in df.columns:
                profile[f'{feature}_mean'] = cluster_data[feature].mean()
                profile[f'{feature}_std'] = cluster_data[feature].std()
        
        # Categorical features
        if 'Contract' in df.columns:
            profile['contract_mode'] = cluster_data['Contract'].mode()[0]
        
        if 'InternetService' in df.columns:
            profile['internet_mode'] = cluster_data['InternetService'].mode()[0]
        
        if 'PaymentMethod' in df.columns:
            profile['payment_mode'] = cluster_data['PaymentMethod'].mode()[0]
        
        # Engineered features (may be present or not)
        if 'tenure_group' in df.columns:
            profile['tenure_group_mode'] = cluster_data['tenure_group'].mode()[0]
        
        if 'high_value_flag' in df.columns:
            profile['high_value_pct'] = cluster_data['high_value_flag'].mean() * 100
        
        if 'auto_payment' in df.columns:
            profile['auto_payment_pct'] = cluster_data['auto_payment'].mean() * 100
        
        if 'family_status' in df.columns:
            profile['family_status_mode'] = cluster_data['family_status'].mode()[0]
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    print(f"\nCluster sizes:")
    for _, row in profiles_df.iterrows():
        print(f"  Cluster {int(row['cluster_id'])}: {int(row['size'])} customers ({row['size_pct']:.1f}%)")
    
    return profiles_df


def assign_cluster_names(profiles_df: pd.DataFrame) -> Dict[int, str]:
    """
    Assign empirical names to clusters based on their characteristics.
    
    Args:
        profiles_df: DataFrame with cluster profiles
        
    Returns:
        Dictionary mapping cluster_id to name
    """
    print_section("Assigning Empirical Cluster Names")
    
    cluster_names = {}
    
    for _, row in profiles_df.iterrows():
        cluster_id = int(row['cluster_id'])
        
        # Extract key characteristics
        tenure_mean = row.get('tenure_mean', 0)
        monthly_charges_mean = row.get('MonthlyCharges_mean', 0)
        churn_rate = row.get('churn_rate', 0)
        service_diversity = row.get('service_diversity_mean', 0)
        high_value_pct = row.get('high_value_pct', 0)
        contract_mode = row.get('contract_mode', 'Month-to-month')
        
        # Build name based on characteristics
        descriptors = []
        
        # Spending level
        if monthly_charges_mean > 80:
            descriptors.append('Premium')
        elif monthly_charges_mean > 55:
            descriptors.append('Standard')
        else:
            descriptors.append('Budget')
        
        # Tenure/Loyalty
        if tenure_mean < 15:
            descriptors.append('New')
        elif tenure_mean < 45:
            descriptors.append('Established')
        else:
            descriptors.append('Loyal')
        
        # Service usage
        if service_diversity > 4:
            descriptors.append('Tech-Savvy')
        elif service_diversity < 2:
            descriptors.append('Basic')
        
        # Churn risk
        if churn_rate > 0.40:
            descriptors.append('High-Risk')
        elif churn_rate < 0.15:
            descriptors.append('Stable')
        
        # Contract type
        if contract_mode == 'Two year':
            descriptors.append('Committed')
        elif contract_mode == 'Month-to-month':
            descriptors.append('Flexible')
        
        # Create unique name
        if len(descriptors) >= 2:
            name = ' '.join(descriptors[:2])
        else:
            name = descriptors[0] if descriptors else f'Segment {cluster_id}'
        
        cluster_names[cluster_id] = name
        print(f"  Cluster {cluster_id}: '{name}'")
        print(f"    (tenure={tenure_mean:.1f}, charges=${monthly_charges_mean:.2f}, "
              f"churn={churn_rate*100:.1f}%, services={service_diversity:.1f})")
    
    # Ensure unique names
    unique_names = set()
    for cluster_id, name in cluster_names.items():
        if name in unique_names:
            # Append cluster number to make unique
            cluster_names[cluster_id] = f"{name} ({cluster_id})"
        unique_names.add(cluster_names[cluster_id])
    
    return cluster_names


def generate_business_actions(profiles_df: pd.DataFrame, 
                              cluster_names: Dict[int, str]) -> pd.DataFrame:
    """
    Generate business actions for each cluster.
    
    Args:
        profiles_df: DataFrame with cluster profiles
        cluster_names: Dictionary mapping cluster_id to name
        
    Returns:
        DataFrame with business actions
    """
    print_section("Generating Business Action Matrix")
    
    actions = []
    
    for _, row in profiles_df.iterrows():
        cluster_id = int(row['cluster_id'])
        name = cluster_names[cluster_id]
        
        tenure_mean = row.get('tenure_mean', 0)
        monthly_charges_mean = row.get('MonthlyCharges_mean', 0)
        churn_rate = row.get('churn_rate', 0)
        service_diversity = row.get('service_diversity_mean', 0)
        contract_mode = row.get('contract_mode', 'Month-to-month')
        
        # Primary insight
        insights = []
        if churn_rate > 0.35:
            insights.append(f"High churn risk ({churn_rate*100:.0f}%)")
        elif churn_rate < 0.15:
            insights.append(f"Low churn risk ({churn_rate*100:.0f}%)")
        
        if tenure_mean < 15:
            insights.append("New customers")
        elif tenure_mean > 50:
            insights.append("Long-tenure loyal customers")
        
        if monthly_charges_mean > 80:
            insights.append("High spenders")
        elif monthly_charges_mean < 40:
            insights.append("Price-sensitive")
        
        if service_diversity > 4:
            insights.append("Heavy service users")
        elif service_diversity < 2:
            insights.append("Basic service users")
        
        primary_insight = "; ".join(insights) if insights else "Mixed characteristics"
        
        # Marketing action
        if churn_rate > 0.35:
            marketing = "Urgent retention campaign with personalized offers"
        elif tenure_mean < 15:
            marketing = "Onboarding program with service education"
        elif monthly_charges_mean > 80:
            marketing = "Premium upselling and loyalty rewards"
        elif service_diversity < 2:
            marketing = "Cross-selling campaign for add-on services"
        else:
            marketing = "Maintain engagement with targeted promotions"
        
        # Retention strategy
        if churn_rate > 0.40:
            retention = "Immediate intervention: dedicated retention team, contract incentives"
        elif churn_rate > 0.25:
            retention = "Proactive outreach: satisfaction surveys, service improvements"
        elif contract_mode == 'Month-to-month':
            retention = "Contract conversion incentives for stability"
        else:
            retention = "Loyalty rewards: referral bonuses, exclusive benefits"
        
        actions.append({
            'cluster_id': cluster_id,
            'cluster_name': name,
            'primary_insight': primary_insight,
            'marketing_action': marketing,
            'retention_strategy': retention
        })
    
    actions_df = pd.DataFrame(actions)
    
    print("\nBusiness Actions Generated:")
    for _, row in actions_df.iterrows():
        print(f"\n  {row['cluster_name']}:")
        print(f"    Insight: {row['primary_insight']}")
        print(f"    Marketing: {row['marketing_action']}")
        print(f"    Retention: {row['retention_strategy']}")
    
    return actions_df


def plot_radar_chart(df: pd.DataFrame, labels: np.ndarray,
                     anova_results: pd.DataFrame, cluster_names: Dict[int, str],
                     figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot radar chart of top 5 discriminating features.
    
    Args:
        df: DataFrame with features
        labels: Cluster labels
        anova_results: DataFrame with ANOVA results
        cluster_names: Dictionary mapping cluster_id to name
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    print_section("Generating Radar Chart")
    
    # Get top 5 features by F-statistic
    top_features = anova_results.head(5)['feature'].tolist()
    print(f"Top 5 discriminating features: {top_features}")
    
    # Compute global min and max for each feature
    global_mins = {}
    global_maxs = {}
    for feature in top_features:
        global_mins[feature] = df[feature].min()
        global_maxs[feature] = df[feature].max()
    
    # Compute cluster means and scale to [0, 1]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    cluster_profiles = {}
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = df[mask]
        
        profile = []
        for feature in top_features:
            mean_val = cluster_data[feature].mean()
            # Scale to [0, 1] using global min-max
            scaled = (mean_val - global_mins[feature]) / (global_maxs[feature] - global_mins[feature] + 1e-10)
            profile.append(scaled)
        
        cluster_profiles[cluster_id] = profile
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(top_features)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Colors for clusters
    colors = COLOR_PALETTE[:n_clusters]
    
    # Plot each cluster
    for i, (cluster_id, profile) in enumerate(cluster_profiles.items()):
        values = profile + profile[:1]  # Complete the circle
        name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=8)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Clusters')
    
    # Title
    ax.set_title('Radar Chart: Top 5 Discriminating Features\n'
                 '(Values scaled to global min-max: 0=global min, 1=global max)',
                 fontsize=12, fontweight='bold', y=1.08)
    
    return save_figure(fig, '23_radar_chart', figures_dir)


def plot_feature_importance(anova_results: pd.DataFrame,
                            figures_dir: Path = FIGURES_DIR) -> str:
    """
    Plot horizontal bar chart of feature importance (F-statistic).
    
    Args:
        anova_results: DataFrame with ANOVA results
        figures_dir: Directory to save figure
        
    Returns:
        Path to saved figure
    """
    print_section("Generating Feature Importance Plot")
    
    # Sort by F-statistic
    sorted_results = anova_results.sort_values('f_statistic', ascending=True)
    
    # Take top 20 features for readability
    top_n = min(20, len(sorted_results))
    plot_data = sorted_results.tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create color gradient based on F-statistic
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(plot_data)))
    
    bars = ax.barh(plot_data['feature'], plot_data['f_statistic'], color=colors, edgecolor='black')
    
    # Add value labels
    for bar, f_stat in zip(bars, plot_data['f_statistic']):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{f_stat:.1f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('ANOVA F-Statistic', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title('Feature Importance: ANOVA F-Statistic by Feature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add significance threshold line
    ax.axvline(x=stats.f.ppf(0.95, len(plot_data)-1, 1000), color='red', 
               linestyle='--', alpha=0.5, label='p<0.05 threshold')
    ax.legend()
    
    return save_figure(fig, '24_feature_importance_bar', figures_dir)


def run_profiling(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
    figures_dir: Path = FIGURES_DIR
) -> Dict[str, Any]:
    """
    Run complete cluster profiling pipeline.
    
    Args:
        df: DataFrame with features (including original columns)
        labels: Cluster labels
        feature_names: List of feature names
        figures_dir: Directory to save figures
        
    Returns:
        Dictionary with all profiling results
    """
    print_header("CLUSTER PROFILING AND INTERPRETATION")
    
    results = {}
    
    # ANOVA F-statistics
    anova_results = compute_anova_f_statistics(df, labels, feature_names)
    results['anova_results'] = anova_results
    
    # Cluster characterization
    profiles_df = characterize_clusters(df, labels, feature_names)
    results['profiles'] = profiles_df
    
    # Assign cluster names
    cluster_names = assign_cluster_names(profiles_df)
    results['cluster_names'] = cluster_names
    
    # Add names to profiles
    profiles_df['cluster_name'] = profiles_df['cluster_id'].map(cluster_names)
    
    # Generate business actions
    actions_df = generate_business_actions(profiles_df, cluster_names)
    results['business_actions'] = actions_df
    
    # Generate plots
    print("\nGenerating profiling plots...")
    radar_path = plot_radar_chart(df, labels, anova_results, cluster_names, figures_dir)
    results['radar_chart_path'] = radar_path
    print(f"  Saved: {radar_path}")
    
    importance_path = plot_feature_importance(anova_results, figures_dir)
    results['feature_importance_path'] = importance_path
    print(f"  Saved: {importance_path}")
    
    return results


def save_profiling_results(results: Dict[str, Any], 
                           profiles_path: Path,
                           actions_path: Path):
    """
    Save profiling results to CSV files.
    
    Args:
        results: Dictionary with profiling results
        profiles_path: Path to save cluster profiles
        actions_path: Path to save business actions
    """
    # Save cluster profiles
    profiles_df = results['profiles']
    profiles_df.to_csv(profiles_path, index=False)
    print(f"\nCluster profiles saved to: {profiles_path}")
    
    # Save business actions
    actions_df = results['business_actions']
    actions_df.to_csv(actions_path, index=False)
    print(f"Business actions saved to: {actions_path}")
