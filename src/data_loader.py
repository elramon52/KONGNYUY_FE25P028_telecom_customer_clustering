"""
Data loading module for Telecom Customer Segmentation.
Handles loading, initial inspection, and basic data cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from .config import DATASET_PATH, TARGET_COL
from .utils import print_header


def load_telecom_data(filepath: Path = DATASET_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and perform initial inspection of telecom customer dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (DataFrame, info_dict)
    """
    print_header("LOADING DATA")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Convert TotalCharges to numeric (coerce errors to NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Basic information
    info_dict = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Print summary
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    print("Missing Values:")
    for col, count in info_dict['missing_values'].items():
        if count > 0:
            pct = info_dict['missing_pct'][col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    
    print(f"\nDuplicate rows: {info_dict['duplicate_rows']}")
       
    # Data types summary
    print("\nData Types:")
    print(f"  Numerical: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"  Categorical: {df.select_dtypes(include=['object']).columns.tolist()}")
    
    # Churn distribution
    if TARGET_COL in df.columns:
        churn_dist = df[TARGET_COL].value_counts()
        churn_pct = df[TARGET_COL].value_counts(normalize=True) * 100
        print(f"\nChurn Distribution:")
        for val in churn_dist.index:
            print(f"  {val}: {churn_dist[val]} ({churn_pct[val]:.1f}%)")
    
    return df, info_dict


def get_column_categories(df: pd.DataFrame) -> Dict[str, list]:
    """
    Categorize columns by type for preprocessing.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column categories
    """
    # Numerical columns
    numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Binary columns (Yes/No or 0/1)
    binary = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
              'PaperlessBilling']
    
    # Categorical columns requiring encoding
    categorical = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    # ID column
    id_col = ['customerID']
    
    # Target column
    target = [TARGET_COL]
    
    return {
        'numerical': numerical,
        'binary': binary,
        'categorical': categorical,
        'id': id_col,
        'target': target
    }


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform detailed data quality inspection.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {}
    
    # Missing values analysis
    missing = df.isnull().sum()
    quality_report['missing_columns'] = missing[missing > 0].to_dict()
    
    # Check for TotalCharges = NaN with tenure = 0
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        zero_tenure = df[df['tenure'] == 0]
        nan_total = df[df['TotalCharges'].isna()]
        quality_report['zero_tenure_count'] = len(zero_tenure)
        quality_report['nan_totalcharges_count'] = len(nan_total)
        quality_report['zero_tenure_with_nan'] = len(zero_tenure[zero_tenure['TotalCharges'].isna()])
    
    # Check for inconsistencies in categorical data
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_vals = df[col].unique()
        quality_report[f'{col}_unique'] = len(unique_vals)
        quality_report[f'{col}_values'] = list(unique_vals)
    
    # Numerical columns statistics
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        if col in df.columns:
            quality_report[f'{col}_stats'] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
    
    return quality_report
