#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import (
    DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH,
    CLUSTER_PROFILES_PATH, BUSINESS_ACTIONS_PATH, FIGURES_DIR,
    RANDOM_STATE, NP_RANDOM_SEED, TARGET_COL
)
from src.utils import (
    print_project_identity, print_header, save_model, 
)
from src.data_loader import load_telecom_data, inspect_data_quality
from src.splitter import stratified_split, verify_split_integrity, get_combined_train_val
from src.eda import run_eda, get_eda_key_findings
from src.preprocessing import TelecomPreprocessor, detect_outliers, preprocess_all_splits
from src.model_selection import run_model_selection
from src.training import (
    run_final_training_and_evaluation, 
    generate_test_evaluation_plots
)
from src.profiling import run_profiling, save_profiling_results


def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(NP_RANDOM_SEED)


def main():
    """Run the complete ML pipeline."""
    
    # ========================================================================
    # PHASE 0: PROJECT IDENTITY
    # ========================================================================
    print_project_identity()
    
    # Set random seeds
    set_random_seeds()
    
    # ========================================================================
    # PHASE 1: DATA LOADING
    # ========================================================================
    df, data_info = load_telecom_data(DATASET_PATH)
    
    # ========================================================================
    # PHASE 2: TRAIN/VALIDATION/TEST SPLIT
    # ========================================================================
    splits = stratified_split(df)
    verify_split_integrity(splits)
    
    # ========================================================================
    # PHASE 3: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    # Run EDA on training set only
    eda_figures = run_eda(splits['train'])
    eda_findings = get_eda_key_findings()
    
    # ========================================================================
    # PHASE 4: DATA PREPROCESSING
    # ========================================================================
    # Outlier analysis (for information only)
    outlier_info = detect_outliers(splits['train'], 
                                    ['tenure', 'MonthlyCharges', 'TotalCharges'])
    
    # Create and fit preprocessor on training set
    preprocessor = TelecomPreprocessor()
    preprocessor.fit(splits['train'])
    
    # Preprocess all splits
    processed_splits = preprocess_all_splits(splits, preprocessor)
    
    # Save preprocessor
    preprocessor.save(PREPROCESSOR_PATH)
    
    # ========================================================================
    # PHASE 5: MODEL SELECTION
    # ========================================================================
    # Prepare feature matrices (exclude ID and target columns)
    feature_cols = preprocessor.get_feature_names()
    
    X_train = processed_splits['train'][feature_cols].values
    X_val = processed_splits['validation'][feature_cols].values
    X_test = processed_splits['test'][feature_cols].values
    
    # Run model selection
    model_selection_results = run_model_selection(X_train, X_val)
    
    # Get selected model parameters
    selected_algorithm = model_selection_results['selected']['algorithm']
    selected_k = model_selection_results['selected']['k']
    selection_details = model_selection_results['selected']['selection_details']
    
    print(f"\nSelected Model: {selected_algorithm} with K={selected_k}")
    
    # ========================================================================
    # PHASE 6: FINAL MODEL TRAINING AND EVALUATION
    # ========================================================================
    training_results = run_final_training_and_evaluation(
        X_train, X_val, X_test,
        splits['test'],  # Original test data for churn validation
        selected_k,
        feature_cols
    )
    
    # Save final model
    save_model(training_results['model'], MODEL_PATH)
    
    # ========================================================================
    # PHASE 7: CLUSTER PROFILING AND INTERPRETATION
    # ========================================================================
    # Combine train and validation for profiling
    df_train_val = get_combined_train_val(splits)
    X_train_val = np.vstack([X_train, X_val])
    
    # Get cluster labels for train+val
    model_final = training_results['model']
    train_val_labels = model_final.predict(X_train_val)
    
    # Run profiling
    profiling_results = run_profiling(
        df_train_val,
        train_val_labels,
        feature_cols
    )
    
    # Save profiling results
    save_profiling_results(
        profiling_results,
        CLUSTER_PROFILES_PATH,
        BUSINESS_ACTIONS_PATH
    )
    
    # ========================================================================
    # PHASE 8: GENERATE TEST EVALUATION PLOTS WITH CLUSTER NAMES
    # ========================================================================
    test_plots = generate_test_evaluation_plots(
        X_test,
        training_results['test_labels'],
        training_results['external_validation'],
        feature_cols,
        profiling_results['cluster_names']
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("PIPELINE EXECUTION COMPLETE")
    
    print("\nGenerated Outputs:")
    print(f"  Models: {MODEL_PATH}")
    print(f"  Preprocessor: {PREPROCESSOR_PATH}")
    print(f"  Cluster Profiles: {CLUSTER_PROFILES_PATH}")
    print(f"  Business Actions: {BUSINESS_ACTIONS_PATH}")
    print(f"  Figures: {FIGURES_DIR}")
    
    print("\nKey Results:")
    print(f"  Selected Algorithm: {selected_algorithm}")
    print(f"  Number of Clusters: {selected_k}")
    print(f"  Test Silhouette Score: {training_results['test_metrics']['silhouette_score']:.4f}")
    print(f"  Test Davies-Bouldin Index: {training_results['test_metrics']['davies_bouldin_index']:.4f}")
    print(f"  Chi-square p-value: {training_results['external_validation']['p_value']:.2e}")
    
    print("\nCluster Names:")
    for cluster_id, name in profiling_results['cluster_names'].items():
        print(f"  Cluster {cluster_id}: {name}")
    
    print("\nNext Steps:")

    print(" Run 'streamlit run dashboard.py' to launch the interactive dashboard")
    
    return {
        'data_info': data_info,
        'splits': splits,
        'eda_figures': eda_figures,
        'eda_findings': eda_findings,
        'model_selection': model_selection_results,
        'training': training_results,
        'profiling': profiling_results,
        'test_plots': test_plots
    }


if __name__ == "__main__":
    results = main()
