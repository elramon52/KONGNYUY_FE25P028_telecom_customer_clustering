"""
Data splitting module for Telecom Customer Segmentation.
Implements stratified train/validation/test split to preserve churn distribution.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from pathlib import Path

from .config import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, RANDOM_STATE, TARGET_COL
from .utils import print_header, print_section


def stratified_split(df: pd.DataFrame, 
                     target_col: str = TARGET_COL,
                     train_ratio: float = TRAIN_RATIO,
                     val_ratio: float = VALIDATION_RATIO,
                     test_ratio: float = TEST_RATIO,
                     random_state: int = RANDOM_STATE) -> Dict[str, pd.DataFrame]:
    """
    Perform stratified train/validation/test split.
    
    The split preserves the original churn rate across all splits.
    First separates test set, then splits remaining into train/validation.
    
    Args:
        df: Input DataFrame
        target_col: Column to use for stratification
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', 'test' DataFrames
    """
    print_header("TRAIN/VALIDATION/TEST SPLIT")
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Convert target to binary for stratification if needed
    if df[target_col].dtype == 'object':
        y_strat = (df[target_col] == 'Yes').astype(int)
    else:
        y_strat = df[target_col]
    
    # Step 1: Split off test set (20%)
    df_temp, df_test = train_test_split(
        df, 
        test_size=test_ratio,
        stratify=y_strat,
        random_state=random_state
    )
    
    # Step 2: Split remaining into train (60%) and validation (20%)
    # Adjusted ratio: val should be 20% of original = 25% of remaining (80%)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    y_temp = y_strat.loc[df_temp.index]
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Create result dictionary
    splits = {
        'train': df_train.reset_index(drop=True),
        'validation': df_val.reset_index(drop=True),
        'test': df_test.reset_index(drop=True)
    }
    
    # Print summary
    print(f"Split completed with stratification on '{target_col}'")
    print(f"\nSplit sizes:")
    total = len(df)
    for name, split_df in splits.items():
        pct = len(split_df) / total * 100
        print(f"  {name.capitalize()}: {len(split_df)} records ({pct:.1f}%)")
    
    # Print churn distribution for each split
    print(f"\nChurn rate by split:")
    overall_churn = (df[target_col] == 'Yes').mean() * 100
    print(f"  Overall: {overall_churn:.1f}%")
    
    for name, split_df in splits.items():
        churn_rate = (split_df[target_col] == 'Yes').mean() * 100
        print(f"  {name.capitalize()}: {churn_rate:.1f}%")
    
    return splits


def verify_split_integrity(splits: Dict[str, pd.DataFrame], 
                           id_col: str = 'customerID') -> bool:
    """
    Verify that splits are non-overlapping and complete.
    
    Args:
        splits: Dictionary with train, validation, test DataFrames
        id_col: Column to use for uniqueness check
        
    Returns:
        True if split is valid
    """
    print_section("Split Integrity Check")
    
    train_ids = set(splits['train'][id_col])
    val_ids = set(splits['validation'][id_col])
    test_ids = set(splits['test'][id_col])
    
    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
    
    if all_overlaps:
        print(f"WARNING: Found {len(all_overlaps)} overlapping IDs!")
        return False
    else:
        print("No overlapping IDs found.")
    
    # Check completeness
    total_original = len(train_ids) + len(val_ids) + len(test_ids)
    total_unique = len(train_ids | val_ids | test_ids)
    
    if total_original == total_unique:
        print(f"All {total_unique} records are unique across splits.")
        return True
    else:
        print(f"WARNING: Mismatch in record counts!")
        return False


def get_combined_train_val(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine training and validation sets for final model training.
    
    Args:
        splits: Dictionary with train, validation, test DataFrames
        
    Returns:
        Combined train+validation DataFrame
    """
    return pd.concat([splits['train'], splits['validation']], 
                     ignore_index=True)


def print_split_summary(splits: Dict[str, pd.DataFrame]):
    """Print detailed summary of each split."""
    print_section("Detailed Split Summary")
    
    for name, df in splits.items():
        print(f"\n{name.upper()} SET:")
        print(f"  Records: {len(df)}")
        print(f"  Features: {df.shape[1]}")
        
        # Numerical summary
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        print(f"  Numerical features (mean):")
        for col in num_cols:
            if col in df.columns:
                print(f"    {col}: {df[col].mean():.2f}")
        
        # Categorical summary
        cat_cols = ['Contract', 'InternetService', 'PaymentMethod']
        print(f"  Top categories:")
        for col in cat_cols:
            if col in df.columns:
                top_val = df[col].value_counts().index[0]
                top_pct = df[col].value_counts().iloc[0] / len(df) * 100
                print(f"    {col}: {top_val} ({top_pct:.1f}%)")
