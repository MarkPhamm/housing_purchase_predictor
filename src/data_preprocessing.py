"""
Data preprocessing module for house purchase prediction.

This module handles data loading, cleaning, feature engineering, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the house purchase dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Check data quality including missing values, duplicates, and data types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict: Dictionary containing data quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict(),
        'target_distribution': df['decision'].value_counts().to_dict()
    }
    
    logger.info(f"Data quality check completed:")
    logger.info(f"  - Shape: {quality_report['shape']}")
    logger.info(f"  - Duplicates: {quality_report['duplicates']}")
    logger.info(f"  - Target distribution: {quality_report['target_distribution']}")
    
    return quality_report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df = df.copy()
    
    # Property age
    df['property_age'] = 2025 - df['constructed_year']
    
    # Price per square foot
    df['price_per_sqft'] = df['price'] / df['property_size_sqft']
    
    # Loan to price ratio
    df['loan_to_price_ratio'] = df['loan_amount'] / df['price']
    
    # Down payment ratio
    df['down_payment_ratio'] = df['down_payment'] / df['price']
    
    # Affordability score (salary to price ratio)
    df['affordability_score'] = df['customer_salary'] / df['price']
    
    # Total rooms
    df['total_rooms'] = df['rooms'] + df['bathrooms']
    
    # Property amenities score
    df['amenities_score'] = df['garage'] + df['garden']
    
    # Risk score (crime + legal cases)
    df['risk_score'] = df['crime_cases_reported'] + df['legal_cases_on_property']
    
    # Monthly payment burden
    df['monthly_payment_burden'] = (df['monthly_expenses'] + 
                                     (df['loan_amount'] / (df['loan_tenure_years'] * 12)))
    
    # Disposable income ratio
    df['disposable_income_ratio'] = (df['customer_salary'] - df['monthly_expenses']) / df['customer_salary']
    
    logger.info(f"Feature engineering completed. New features created: {df.shape[1] - len(df.columns) + 10}")
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                  categorical_cols: List[str],
                                  encoders: Dict = None,
                                  fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (List[str]): List of categorical column names
        encoders (Dict): Dictionary of fitted encoders (for test set)
        fit (bool): Whether to fit new encoders or use existing ones
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Encoded dataframe and dictionary of encoders
    """
    df = df.copy()
    
    if fit:
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        logger.info(f"Categorical features encoded: {categorical_cols}")
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    return df, encoders


def split_data(df: pd.DataFrame, 
               target_col: str = 'decision',
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                   pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of training data for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Separate features and target
    X = df.drop([target_col, 'property_id'], axis=1, errors='ignore')
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  - Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  - Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  - Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   X_test: pd.DataFrame,
                   numerical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        numerical_cols (List[str]): List of numerical column names
        
    Returns:
        Tuple: Scaled X_train, X_val, X_test, and fitted scaler
    """
    scaler = StandardScaler()
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # Fit on train, transform all sets
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    logger.info(f"Feature scaling completed for {len(numerical_cols)} numerical features")
    
    return X_train, X_val, X_test, scaler


def preprocess_pipeline(filepath: str,
                         random_state: int = 42) -> Tuple:
    """
    Complete preprocessing pipeline from raw data to train/val/test sets.
    
    Args:
        filepath (str): Path to the CSV file
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test, encoders, scaler, feature_names
    """
    # Load data
    df = load_data(filepath)
    
    # Check data quality
    quality_report = check_data_quality(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Define categorical and numerical columns
    categorical_cols = ['country', 'city', 'property_type', 'furnishing_status']
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df, categorical_cols, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_col='decision', random_state=random_state
    )
    
    # Get numerical columns (all columns after encoding)
    numerical_cols = X_train.columns.tolist()
    
    # Scale features
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test, numerical_cols
    )
    
    feature_names = X_train.columns.tolist()
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders, scaler, feature_names

