"""
Data preprocessing module for Telecom Customer Segmentation.
Implements feature engineering, encoding, and scaling with data leakage prevention.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle

from .config import (
    TENURE_GROUP_BINS, TENURE_GROUP_LABELS, CONTRACT_MAPPING,
    AUTO_PAYMENT_METHODS, SERVICE_COLUMNS, CATEGORICAL_COLUMNS,
    BINARY_COLUMNS, NUMERICAL_COLUMNS, RANDOM_STATE
)
from .utils import print_header, print_section


class TelecomPreprocessor:
    """
    Custom preprocessor for telecom customer data.
    
    Implements:
    - Missing value imputation (TotalCharges for tenure=0)
    - Feature engineering (avg_monthly_spend, tenure_group, etc.)
    - Categorical encoding (one-hot and ordinal)
    - Feature scaling (StandardScaler)
    
    All statistics computed on training set only to prevent data leakage.
    """
    
    def __init__(self):
        """Initialize preprocessor with empty state."""
        # Scalers (fit on training set)
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Feature names (stored after fit)
        self.feature_names_ = None
        
        # Training set statistics
        self.high_value_monthly_threshold = None
        self.high_value_tenure_threshold = None
        
        # One-hot encoded column names
        self.onehot_columns = []
        
    def fit(self, df: pd.DataFrame) -> 'TelecomPreprocessor':
        """
        Fit preprocessor on training data.
        Computes all statistics needed for transformation.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        print_section("Fitting Preprocessor on Training Set")
        
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self._impute_missing_values(df_processed)
        
        # Step 2: Compute thresholds for high_value_flag (on training set)
        self.high_value_monthly_threshold = df_processed['MonthlyCharges'].quantile(0.75)
        self.high_value_tenure_threshold = df_processed['tenure'].quantile(0.75)
        print(f"  High-value thresholds computed:")
        print(f"    MonthlyCharges > {self.high_value_monthly_threshold:.2f}")
        print(f"    tenure > {self.high_value_tenure_threshold:.2f}")
        
        # Step 3: Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Step 4: Encode categorical variables (get column names)
        df_processed = self._encode_categorical(df_processed, fit=True)
        
        # Step 5: Encode binary variables
        df_processed = self._encode_binary(df_processed)
        
        # Step 6: Fit scaler on numerical features
        numerical_features = df_processed[NUMERICAL_COLUMNS].copy()
        self.scaler.fit(numerical_features)
        self.scaler_fitted = True
        print(f"  StandardScaler fitted on {len(NUMERICAL_COLUMNS)} numerical features")
        
        # Store final feature names (excluding ID and target)
        exclude_cols = ['customerID', 'Churn']
        self.feature_names_ = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"  Total features after preprocessing: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame with consistent columns
        """
        if not self.scaler_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self._impute_missing_values(df_processed)
        
        # Step 2: Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Step 3: Encode categorical variables
        df_processed = self._encode_categorical(df_processed, fit=False)
        
        # Step 4: Encode binary variables
        df_processed = self._encode_binary(df_processed)
        
        # Step 5: Scale numerical features
        numerical_features = df_processed[NUMERICAL_COLUMNS].copy()
        scaled_values = self.scaler.transform(numerical_features)
        for i, col in enumerate(NUMERICAL_COLUMNS):
            df_processed[col] = scaled_values[:, i]
        
        # Step 6: Ensure consistent columns
        df_processed = self._ensure_columns(df_processed)
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing TotalCharges values.
        For customers with tenure=0, set TotalCharges = MonthlyCharges (first month charge).
        """
        df = df.copy()
        
        # Find rows with missing TotalCharges or tenure=0
        mask = (df['TotalCharges'].isna()) | (df['tenure'] == 0)
        
        # Impute with MonthlyCharges
        df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on domain knowledge.
        """
        df = df.copy()
        
        # 1. avg_monthly_spend: TotalCharges / (tenure + 1)
        df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # 2. tenure_group: Bin tenure into New, Mid, Loyal
        df['tenure_group'] = pd.cut(
            df['tenure'], 
            bins=TENURE_GROUP_BINS, 
            labels=TENURE_GROUP_LABELS,
            include_lowest=True
        )
        # Convert to ordinal (0, 1, 2)
        df['tenure_group'] = df['tenure_group'].map({'New': 0, 'Mid': 1, 'Loyal': 2})
        
        # 3. service_diversity: Count of "Yes" in add-on services
        service_df = df[SERVICE_COLUMNS].copy()
        for col in SERVICE_COLUMNS:
            service_df[col] = (service_df[col] == 'Yes').astype(int)
        df['service_diversity'] = service_df.sum(axis=1)
        
        # 4. contract_commitment: Ordinal encoding of Contract
        df['contract_commitment'] = df['Contract'].map(CONTRACT_MAPPING)
        df = df.drop(columns=['Contract'])  # Drop original
        
        # 5. auto_payment: 1 if automatic payment method
        df['auto_payment'] = df['PaymentMethod'].isin(AUTO_PAYMENT_METHODS).astype(int)
        
        # 6. high_value_flag: 1 if high monthly charges AND long tenure
        df['high_value_flag'] = (
            (df['MonthlyCharges'] > self.high_value_monthly_threshold) & 
            (df['tenure'] > self.high_value_tenure_threshold)
        ).astype(int)
        
        # 7. family_status: Combination of Partner and Dependents
        # 0 = single/no dependents, 1 = with partner only, 2 = with dependents only, 3 = both
        partner = (df['Partner'] == 'Yes').astype(int)
        dependents = (df['Dependents'] == 'Yes').astype(int)
        df['family_status'] = partner + 2 * dependents
        # Note: Partner and Dependents are kept for binary encoding
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical variables.
        Uses drop_first=True to avoid multicollinearity.
        """
        df = df.copy()
        
        # Columns to one-hot encode
        cols_to_encode = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'PaymentMethod']
        
        for col in cols_to_encode:
            if col in df.columns:
                # Create dummies
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                if fit:
                    # Store column names during fit
                    self.onehot_columns.extend(dummies.columns.tolist())
                
                # Concatenate and drop original
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode binary columns to 0/1.
        """
        df = df.copy()
        
        binary_mappings = {
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # SeniorCitizen is already 0/1
        
        return df
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has exactly the same columns as during fit.
        Add missing columns as zeros, drop extra columns.
        """
        # Keep only feature columns (exclude ID and target)
        exclude_cols = ['customerID', 'Churn']
        
        # Add missing columns as zeros
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the required columns in the correct order
        result_cols = self.feature_names_ + [col for col in ['customerID', 'Churn'] if col in df.columns]
        df = df[result_cols]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after preprocessing."""
        return self.feature_names_
    
    def save(self, filepath: Path):
        """Save preprocessor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'TelecomPreprocessor':
        """Load preprocessor from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def detect_outliers(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
    """
    Detect outliers using IQR method.
    For information only - outliers are kept as they represent real customers.
    
    Args:
        df: DataFrame
        columns: Columns to check for outliers
        
    Returns:
        Dictionary with outlier information for each column
    """
    print_section("Outlier Analysis (IQR Method)")
    
    outlier_info = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min_outlier': outliers[col].min() if len(outliers) > 0 else None,
            'max_outlier': outliers[col].max() if len(outliers) > 0 else None
        }
        
        print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
        print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    print("\nNote: Outliers are kept as they represent legitimate customer behaviors.")
    
    return outlier_info


def preprocess_all_splits(splits: Dict[str, pd.DataFrame], 
                          preprocessor: TelecomPreprocessor) -> Dict[str, pd.DataFrame]:
    """
    Preprocess all data splits using fitted preprocessor.
    
    Args:
        splits: Dictionary with train, validation, test DataFrames
        preprocessor: Fitted TelecomPreprocessor
        
    Returns:
        Dictionary with preprocessed DataFrames
    """
    print_header("PREPROCESSING ALL SPLITS")
    
    processed_splits = {}
    
    for name, df in splits.items():
        print(f"\nProcessing {name} set...")
        processed = preprocessor.transform(df)
        processed_splits[name] = processed
        print(f"  Shape: {processed.shape}")
    
    # Print summary
    original_features = len([c for c in splits['train'].columns if c not in ['customerID', 'Churn']])
    processed_features = len(preprocessor.feature_names_)
    new_features = processed_features - original_features
    
    print(f"\nPreprocessing Summary:")
    print(f"  Original features: {original_features}")
    print(f"  Processed features: {processed_features}")
    print(f"  New engineered features: {new_features}")
    
    return processed_splits
