"""
Prediction module for house purchase prediction.

This module loads a trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePurchasePredictor:
    """
    A class for making predictions on house purchase decisions.
    """
    
    def __init__(self, model_path: str, scaler_path: str, 
                 encoders_path: str, feature_names_path: str):
        """
        Initialize the predictor with trained model and preprocessing artifacts.
        
        Args:
            model_path (str): Path to the trained model file
            scaler_path (str): Path to the fitted scaler
            encoders_path (str): Path to the encoders
            feature_names_path (str): Path to the feature names JSON
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info("Model and artifacts loaded successfully")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to input data.
        
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
        
        # Affordability score
        df['affordability_score'] = df['customer_salary'] / df['price']
        
        # Total rooms
        df['total_rooms'] = df['rooms'] + df['bathrooms']
        
        # Property amenities score
        df['amenities_score'] = df['garage'] + df['garden']
        
        # Risk score
        df['risk_score'] = df['crime_cases_reported'] + df['legal_cases_on_property']
        
        # Monthly payment burden
        df['monthly_payment_burden'] = (df['monthly_expenses'] + 
                                         (df['loan_amount'] / (df['loan_tenure_years'] * 12)))
        
        # Disposable income ratio
        df['disposable_income_ratio'] = (df['customer_salary'] - df['monthly_expenses']) / df['customer_salary']
        
        return df
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical features
        categorical_cols = ['country', 'city', 'property_type', 'furnishing_status']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        # Drop property_id if present
        if 'property_id' in df.columns:
            df = df.drop('property_id', axis=1)
        
        # Drop decision column if present (target)
        if 'decision' in df.columns:
            df = df.drop('decision', axis=1)
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in input data, setting to 0")
                df[feature] = 0
        
        # Select only the features used in training
        df = df[self.feature_names]
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df_scaled
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Predicted classes (0 or 1)
        """
        df_processed = self.preprocess_input(df)
        predictions = self.model.predict(df_processed)
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        df_processed = self.preprocess_input(df)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df_processed)
            return probabilities
        else:
            logger.warning("Model does not support probability predictions")
            return None
    
    def predict_with_explanation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with probability scores and explanations.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities
        """
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['Buy' if p == 1 else 'Not Buy' for p in predictions]
        })
        
        if probabilities is not None:
            results['probability_not_buy'] = probabilities[:, 0]
            results['probability_buy'] = probabilities[:, 1]
            results['confidence'] = np.max(probabilities, axis=1)
        
        return results


def load_and_predict(data_path: str, 
                     model_path: str,
                     scaler_path: str,
                     encoders_path: str,
                     feature_names_path: str) -> pd.DataFrame:
    """
    Load data, make predictions, and return results.
    
    Args:
        data_path (str): Path to the CSV file with input data
        model_path (str): Path to the trained model
        scaler_path (str): Path to the scaler
        encoders_path (str): Path to the encoders
        feature_names_path (str): Path to feature names
        
    Returns:
        pd.DataFrame: Predictions with probabilities
    """
    # Load predictor
    predictor = HousePurchasePredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        encoders_path=encoders_path,
        feature_names_path=feature_names_path
    )
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Make predictions
    logger.info("Making predictions...")
    results = predictor.predict_with_explanation(df)
    
    # Combine with original data
    output = pd.concat([df, results], axis=1)
    
    logger.info(f"Predictions completed for {len(output)} samples")
    logger.info(f"Predicted Buy: {(results['prediction'] == 1).sum()}")
    logger.info(f"Predicted Not Buy: {(results['prediction'] == 0).sum()}")
    
    return output


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_file.csv>")
        print("Note: Make sure model artifacts are in ../models/ directory")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Find the latest model files in models directory
    models_dir = Path("../models")
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if not model_files:
        print("No trained model found. Please run main_pipeline.py first.")
        sys.exit(1)
    
    # Get the latest model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest_model.stem.split('_')[-1]
    
    model_path = str(latest_model)
    scaler_path = str(models_dir / f"scaler_{timestamp}.pkl")
    encoders_path = str(models_dir / f"encoders_{timestamp}.pkl")
    feature_names_path = str(models_dir / f"feature_names_{timestamp}.json")
    
    # Make predictions
    results = load_and_predict(
        data_path=data_file,
        model_path=model_path,
        scaler_path=scaler_path,
        encoders_path=encoders_path,
        feature_names_path=feature_names_path
    )
    
    # Save results
    output_file = data_file.replace('.csv', '_predictions.csv')
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

